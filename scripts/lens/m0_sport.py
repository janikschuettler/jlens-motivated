"""M0.3 (lens env): "Think of a sport, then name it" — is a sport at the top of
the J-lens at the colon? Reports J-lens and logit-lens top-25 at every workspace
layer, for a chat-template variant (the pass criterion) and a raw-text variant.

uv run --project envs/lens python scripts/lens/m0_sport.py
"""

import torch

from jmr import config
from jmr.checks import Checks
from jmr.lens.load import lens_file, load_lens, load_model
from jmr.lens.readout import readout, residuals
from jmr.tokens import chat_text

SPORTS = {
    "soccer", "football", "basketball", "tennis", "baseball", "cricket", "golf", "hockey",
    "rugby", "swimming", "volleyball", "badminton", "boxing", "cycling", "running", "skiing",
    "surfing", "wrestling", "softball", "lacrosse", "handball", "archery", "fencing", "rowing",
    "sailing", "gymnastics", "karate", "judo", "taekwondo", "skating", "snowboarding",
    "climbing", "polo", "squash", "bowling", "darts", "snooker", "billiards", "marathon",
    "athletics", "triathlon", "ping", "pingpong", "table", "lacrosse", "curling", "diving",
    "kayaking", "canoeing", "badminton", "netball", "sumo",
}


def is_sport(token: str) -> bool:
    word = token.strip().lower()
    if word in SPORTS:
        return True
    return len(word) >= 4 and word.isalpha() and any(s.startswith(word) for s in SPORTS)


def main() -> None:
    checks = Checks("m0_3_sport")
    lens_cfg = config.load("lens")
    layers = lens_cfg["workspace_layers"]
    hf_model, tokenizer, lens_model = load_model()
    lens = load_lens(lens_file())

    user = "Think of a sport, then name it."
    variants = {
        "chat": chat_text(tokenizer, [{"role": "user", "content": user}], enable_thinking=False) + "Sport:",
        "raw": "Think of a sport, then name it:",
    }
    results = {}
    for name, text in variants.items():
        ids = tokenizer(text, add_special_tokens=False).input_ids
        checks.check(f"{name}: last token is ':'", tokenizer.decode(ids[-1:]) == ":", repr(tokenizer.decode(ids[-3:])))
        resid = residuals(lens_model, ids, layers)
        last = {layer: r[-1:] for layer, r in resid.items()}
        tables = {}
        for lens_name, which in (("jlens", lens), ("logit_lens", None)):
            out = readout(lens_model, last, lens=which, top_k=lens_cfg["top_k"])
            tables[lens_name] = {
                layer: [tokenizer.decode([t]) for t in out["top_ids"][li][0]] for li, layer in enumerate(layers)
            }
        with torch.no_grad():
            gen = hf_model.generate(torch.tensor([ids], device=lens_model.input_device), max_new_tokens=6, do_sample=False)
        named = tokenizer.decode(gen[0, len(ids):])

        jl = tables["jlens"]
        rank1 = [layer for layer in layers if is_sport(jl[layer][0])]
        top10 = [layer for layer in layers if any(is_sport(t) for t in jl[layer][:10])]
        results[name] = {"tables": tables, "named": named, "rank1_layers": rank1, "top10_layers": top10}
        print(f"\n== {name}: model names {named!r}")
        for layer in layers:
            print(f"  L{layer:2d} jlens: {jl[layer][:8]}")
            print(f"      logit: {tables['logit_lens'][layer][:8]}")

    chat = results["chat"]
    checks.check("chat: sport at rank 1 in ≥ 1 workspace layer", len(chat["rank1_layers"]) >= 1, chat["rank1_layers"])
    checks.check("chat: sport in top 10 in ≥ 5 of 10 workspace layers", len(chat["top10_layers"]) >= 5, chat["top10_layers"])
    checks.info["results"] = results
    checks.info["raw_variant_rank1_layers"] = results["raw"]["rank1_layers"]
    checks.info["raw_variant_top10_layers"] = results["raw"]["top10_layers"]
    checks.finish()


if __name__ == "__main__":
    main()
