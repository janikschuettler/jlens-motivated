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
    kwargs = dict(
        model=model_cfg["hf_id"],
        revision=model_cfg["revision"],
        tokenizer_revision=model_cfg["revision"],
        dtype=model_cfg["dtype"],
        gpu_memory_utilization=vllm_cfg["gpu_memory_utilization"],
        max_model_len=vllm_cfg["max_model_len"],
        logprobs_mode=vllm_cfg["logprobs_mode"],
        seed=sampling_cfg["base_seed"],
    )
    # A model profile may override engine settings (Gemma-3-27B: cpu_offload_gb, enforce_eager, ...).
    kwargs.update(model_cfg.get("vllm", {}))
    llm = LLM(**kwargs)
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


# Optional SamplingParams fields a sampling config may set (absent = vLLM's default).
OPTIONAL_SAMPLING_KEYS = ("repetition_penalty", "presence_penalty", "frequency_penalty")


def sampling_params(cfg: dict, seed: int, *, max_tokens: int | None = None) -> SamplingParams:
    extra = {k: cfg[k] for k in OPTIONAL_SAMPLING_KEYS if cfg.get(k) is not None}
    return SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        min_p=cfg["min_p"],
        max_tokens=max_tokens or cfg["max_tokens"],
        seed=seed,
        logprobs=1,  # the sampled token's log-prob, plus the top-1
        skip_special_tokens=False,
        **extra,
    )


@dataclass
class Request:
    """One generation from prebuilt chat messages. `sampling` is a full sampling
    config (temperature, top_p, top_k, min_p, max_tokens, optional penalties);
    it is recorded in the run as given."""

    run_id: str
    run_set: str
    question: Question
    condition: str
    hint_letter: str | None
    messages: list[dict]
    enable_thinking: bool
    sampling: dict
    seed: int
    sample_idx: int = 0


def generate_requests(engine: Engine, requests: list[Request]) -> tuple[list[RunRecord], dict]:
    """Generate one completion per request in a single vLLM call (per-request
    sampling params), and return run records and throughput stats."""
    ids_list = [encode_chat(engine.tokenizer, r.messages, enable_thinking=r.enable_thinking) for r in requests]
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in ids_list]
    params = [sampling_params(r.sampling, r.seed) for r in requests]
    start = time.time()
    outputs = engine.llm.generate(prompts, params) if requests else []
    elapsed = time.time() - start

    records = []
    for req, ids, out in zip(requests, ids_list, outputs):
        if list(out.prompt_token_ids) != ids:
            raise RuntimeError(f"{req.run_id}: vLLM's prompt token IDs differ from the ones sent")
        completion = out.outputs[0]
        token_ids = list(completion.token_ids)
        logprobs = [step[token].logprob for step, token in zip(completion.logprobs, token_ids)]
        records.append(
            RunRecord(
                run_id=req.run_id,
                run_set=req.run_set,
                question_id=req.question.id,
                dataset=req.question.dataset,
                condition=req.condition,
                hint_letter=req.hint_letter,
                correct_letter=req.question.correct,
                letters=list(req.question.letters),
                messages=req.messages,
                prompt_token_ids=ids,
                completion_token_ids=token_ids,
                completion_logprobs=logprobs,
                completion_text=completion.text,
                finish_reason=completion.finish_reason,
                seed=req.seed,
                model={"hf_id": engine.model_cfg["hf_id"], "revision": engine.model_cfg["revision"]},
                tokenizer_fingerprint=engine.tokenizer_fingerprint,
                sampling=req.sampling,
                enable_thinking=req.enable_thinking,
                sample_idx=req.sample_idx,
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
    requests = []
    for question in questions:
        for condition in conditions:
            messages, hint = build_messages(question, condition)
            n_samples = samples[question.id] if isinstance(samples, dict) else samples
            for k in range(n_samples):
                run_id = f"{run_set}/{question.id}/{condition}{'/' + arm if arm else ''}/{k}"
                seed = run_seed(engine.sampling_cfg["base_seed"], run_id)
                requests.append(Request(run_id, run_set, question, condition, hint, messages,
                                        enable_thinking, sampling_cfg, seed, k))
    return generate_requests(engine, requests)
