"""M0.1 (gen env): load Qwen3-8B in vLLM, record the KV-cache size and tokenizer
fingerprint, and check that one thinking-mode generation has <think>…</think>
followed by an answer. Also records whether <think> is in the prompt.

uv run --project envs/gen python scripts/gen/m0_check_generation.py
"""

import vllm

from jmr import config
from jmr.checks import Checks
from jmr.gen.generate import build_engine, kv_cache_tokens, sampling_params
from jmr.tokens import encode_chat

QUESTION = (
    "Which gas do plants absorb from the air for photosynthesis?\n\n(A) Oxygen\n(B) Carbon dioxide\n"
    "(C) Nitrogen\n(D) Helium\n\nThink it through, then give your final answer on the last line "
    'in the form "Answer: X", where X is the letter of your choice.'
)


def main() -> None:
    checks = Checks("m0_1_model_gen_env")
    model_cfg = config.load("model")
    engine = build_engine()
    tok = engine.tokenizer
    checks.info["vllm"] = vllm.__version__
    checks.info["transformers"] = __import__("transformers").__version__
    checks.info["tokenizer_fingerprint"] = engine.tokenizer_fingerprint
    checks.info["kv_cache_tokens"] = kv_cache_tokens(engine)

    think_start = tok.convert_tokens_to_ids(model_cfg["special_tokens"]["think_start"])
    think_end = tok.convert_tokens_to_ids(model_cfg["special_tokens"]["think_end"])
    checks.info["think_token_ids"] = {"think_start": think_start, "think_end": think_end}

    ids = encode_chat(tok, [{"role": "user", "content": QUESTION}], enable_thinking=True)
    checks.info["think_start_in_prompt"] = think_start in ids
    checks.info["prompt_tail"] = tok.decode(ids[-8:])
    params = sampling_params(engine.sampling_cfg, seed=1234, max_tokens=4096)
    out = engine.llm.generate([vllm.TokensPrompt(prompt_token_ids=ids)], params)[0].outputs[0]
    completion = list(out.token_ids)
    checks.info["finish_reason"] = out.finish_reason
    checks.info["completion_tokens"] = len(completion)
    checks.info["completion_text"] = out.text

    has_start = think_start in ids or think_start in completion
    checks.check("<think> present (prompt or completion)", has_start,
                 "in prompt" if think_start in ids else "generated")
    checks.check("</think> generated", think_end in completion)
    if think_end in completion:
        response = tok.decode(completion[completion.index(think_end) + 1 :])
        checks.check("answer after </think>", "Answer:" in response, response.strip()[-80:])
    checks.check("generation stopped by itself", out.finish_reason == "stop", out.finish_reason)
    checks.finish()


if __name__ == "__main__":
    main()
