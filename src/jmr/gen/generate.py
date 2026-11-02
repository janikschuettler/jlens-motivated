"""vLLM generation → run records. gen env only."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

from jmr import config
from jmr.prompts import Question, build_messages
from jmr.records import RunRecord
from jmr.tokens import encode_chat, fingerprint


@dataclass
class Engine:
    llm: LLM
    tokenizer: object
    tokenizer_fingerprint: str
    model_cfg: dict
    sampling_cfg: dict


def build_engine() -> Engine:
    model_cfg = config.load("model")
    sampling_cfg = config.load("sampling")
    vllm_cfg = sampling_cfg["vllm"]
    llm = LLM(
        model=model_cfg["hf_id"],
        revision=model_cfg["revision"],
        tokenizer_revision=model_cfg["revision"],
        dtype=model_cfg["dtype"],
        gpu_memory_utilization=vllm_cfg["gpu_memory_utilization"],
        max_model_len=vllm_cfg["max_model_len"],
        logprobs_mode=vllm_cfg["logprobs_mode"],
        seed=sampling_cfg["base_seed"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"], revision=model_cfg["revision"])
    if fingerprint(tokenizer) != fingerprint(llm.get_tokenizer()):
        raise RuntimeError("vLLM's tokenizer and AutoTokenizer tokenize differently")
    return Engine(llm, tokenizer, fingerprint(tokenizer), model_cfg, sampling_cfg)


def kv_cache_tokens(engine: Engine) -> int | None:
    try:
        cache = engine.llm.llm_engine.vllm_config.cache_config
        return cache.num_gpu_blocks * cache.block_size
    except (AttributeError, TypeError):
        return None


def run_seed(base_seed: int, run_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{run_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def sampling_params(cfg: dict, seed: int, *, max_tokens: int | None = None) -> SamplingParams:
    return SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        min_p=cfg["min_p"],
        max_tokens=max_tokens or cfg["max_tokens"],
        seed=seed,
        logprobs=1,  # the sampled token's log-prob, plus the top-1
        skip_special_tokens=False,
    )


def generate_runs(
    engine: Engine,
    questions: list[Question],
    conditions: list[str],
    run_set: str,
    *,
    samples: int | dict[str, int] = 1,
    max_tokens: int | None = None,
    overrides: dict | None = None,
    arm: str = "",
) -> tuple[list[RunRecord], dict]:
    """Generate `samples` completions per question x condition (an int, or a dict
    from question ID to count), each with its own seed. `overrides` replaces
    sampling settings (for example the temperature) and is recorded in each run;
    `arm` is added to the run ID so different arms get different seeds."""
    enable_thinking = engine.model_cfg["enable_thinking"]
    sampling_cfg = {k: v for k, v in engine.sampling_cfg.items() if k != "vllm"}
    if max_tokens:
        sampling_cfg["max_tokens"] = max_tokens
    sampling_cfg.update(overrides or {})
    pending = []
    for question in questions:
        for condition in conditions:
            messages, hint = build_messages(question, condition)
            ids = encode_chat(engine.tokenizer, messages, enable_thinking=enable_thinking)
            n_samples = samples[question.id] if isinstance(samples, dict) else samples
            for k in range(n_samples):
                run_id = f"{run_set}/{question.id}/{condition}{'/' + arm if arm else ''}/{k}"
                seed = run_seed(engine.sampling_cfg["base_seed"], run_id)
                pending.append((run_id, question, condition, hint, messages, ids, seed, k))

    prompts = [TokensPrompt(prompt_token_ids=p[5]) for p in pending]
    params = [sampling_params(sampling_cfg, p[6]) for p in pending]
    start = time.time()
    outputs = engine.llm.generate(prompts, params)
    elapsed = time.time() - start

    records = []
    for (run_id, question, condition, hint, messages, ids, seed, k), out in zip(pending, outputs):
        if list(out.prompt_token_ids) != ids:
            raise RuntimeError(f"{run_id}: vLLM's prompt token IDs differ from the ones sent")
        completion = out.outputs[0]
        token_ids = list(completion.token_ids)
        logprobs = [step[token].logprob for step, token in zip(completion.logprobs, token_ids)]
        records.append(
            RunRecord(
                run_id=run_id,
                run_set=run_set,
                question_id=question.id,
                dataset=question.dataset,
                condition=condition,
                hint_letter=hint,
                correct_letter=question.correct,
                letters=list(question.letters),
                messages=messages,
                prompt_token_ids=ids,
                completion_token_ids=token_ids,
                completion_logprobs=logprobs,
                completion_text=completion.text,
                finish_reason=completion.finish_reason,
                seed=seed,
                model={"hf_id": engine.model_cfg["hf_id"], "revision": engine.model_cfg["revision"]},
                tokenizer_fingerprint=engine.tokenizer_fingerprint,
                sampling=sampling_cfg,
                enable_thinking=enable_thinking,
                sample_idx=k,
            )
        )
    generated = sum(len(r.completion_token_ids) for r in records)
    stats = {
        "n_runs": len(records),
        "elapsed_s": round(elapsed, 1),
        "generated_tokens": generated,
        "tokens_per_s": round(generated / elapsed, 1) if elapsed else None,
        "mean_completion_tokens": round(generated / len(records), 1) if records else None,
    }
    return records, stats
