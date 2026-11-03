"""Assemble docs/overnight/data/models_grader.json (derived numbers only) for the models x
grader-bug report and figure.

  python scripts/explore/mg_build_data.py

Inputs (all derived, data/ is gitignored):
  data/models_grader/summary.json        (scripts/explore/analyze_or_grader.py; API screen)
  data/models_grader/judge.jsonl         (LLM labels of followers; only counts are copied)
  data/models_grader/by_eye.json         (hand labels of the non-open followers; counts copied)
  data/models_grader/cost_log.jsonl      (per-batch usage and cost)
  data/models_grader/local/*_rates.json  (scripts/explore/mg_local_rates.py; local Qwen3-8B/14B)
  data/models_grader/local/*_lens.json   (scripts/lens/analyze_models_grader.py), if present
  configs/explore/models_grader.yaml     (model IDs, providers)
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/models_grader"
CFG = yaml.safe_load(open(REPO / "configs/explore/models_grader.yaml"))
LABEL = {"gpt-oss-20b": "gpt-oss-20b (API)", "qwen3-14b": "Qwen3-14B (API)", "qwen3-32b": "Qwen3-32B",
         "qwen3.6-27b": "Qwen3.6-27B", "deepseek-v4-flash": "DeepSeek-V4-Flash", "gemma-3-12b-it": "Gemma-3-12B-it",
         "gemma-3-27b-it": "Gemma-3-27B-it", "gemma-4-31b-it": "Gemma-4-31B-it", "llama-3.1-8b-it": "Llama-3.1-8B-it",
         "llama-3.3-70b-it": "Llama-3.3-70B-it", "qwen8b_local": "Qwen3-8B (local)", "qwen14b_local": "Qwen3-14B (local)",
         "gptoss_local": "gpt-oss-20b (local)", "nt8_local": "Qwen3-8B (local, no thinking)",
         "nt14_local": "Qwen3-14B (local, no thinking)", "qwen2.5-7b-it": "Qwen2.5-7B-it"}
ORDER = [("Fit on our L40S (local generation + lens possible)",
          ["qwen8b_local", "nt8_local", "qwen14b_local", "nt14_local", "qwen3-14b", "gptoss_local", "gpt-oss-20b", "gemma-3-12b-it", "llama-3.1-8b-it", "qwen2.5-7b-it"]),
         ("API only (too large for the L40S in bf16)",
          ["qwen3.6-27b", "gemma-3-27b-it", "gemma-4-31b-it", "qwen3-32b", "llama-3.3-70b-it", "deepseek-v4-flash"])]


def trip(p):
    return None if p is None or p[0] is None else {"v": round(p[0], 4), "lo": round(p[1], 4), "hi": round(p[2], 4)}


def main():
    S = json.load(open(D / "summary.json"))
    out = {"description": "models x grader-bug screen, derived numbers (docs/overnight/reports/models_grader.md)",
           "rates": {}, "figure_order": [], "judge_counts": {}, "by_eye": None, "cost": {}, "lens": {}}
    for m, ms in S["models"].items():
        mc = CFG["models"][m]
        eps = D / m / "endpoints.json"
        prov = mc.get("provider", {}).get("order", [None])[0]
        quant = None
        if eps.exists():
            for e in json.load(open(eps)):
                if e["provider_name"] == prov:
                    quant = e["quantization"]
        out["rates"][m] = {
            "label": LABEL[m], "sub": f"{ms['n_known']} items", "or_id": mc["id"], "provider": prov, "quantization": quant,
            "lens": mc.get("lens"), "fits_l40s": mc.get("fits_l40s"), "n_known": ms["n_known"],
            "n_screen_items": ms["n_screen_items"], "screen_acc": round(ms["screen_acc"], 4), "cost_usd": ms["cost"],
            "arms": {a: {**(trip(s["p_B"]) or {}), "n": s["runs"], "items": s["items"], "n_B": s["n_B"],
                         "p_correct": trip(s["p_correct"]), "p_none": trip(s["p_none"]), "n_trunc": s["n_trunc"],
                         "mean_tokens": s["mean_tokens"], "kw_labels": s["kw_labels"],
                         **({"delta_vs_null": trip(v)} if (v := next((x for k, x in s.items() if k.startswith("delta_B")), None)) else {})}
                     for a, s in ms["arms"].items()}}
    for name, f in (("qwen8b_local", "qwen8b_rates.json"), ("qwen14b_local", "qwen14b_rates.json"),
                    ("nt8_local", "nt8_rates.json"), ("nt14_local", "nt14_rates.json")):
        p = D / "local" / f
        if p.exists():
            L = json.load(open(p))
            n_items = max(v["items"] for v in L.values())
            out["rates"][name] = {"label": LABEL[name], "sub": f"{n_items} items", "local": True,
                                  "arms": {a: {**v["p_B"], "n": v["runs"], "items": v["items"], "n_B": v["n_B"],
                                               "p_none": v["p_none"], "n_trunc": v["n_trunc"],
                                               "follower_grader_outcome": v["follower_grader_outcome"],
                                               "mean_tokens": v["mean_tokens"]} for a, v in L.items()}}
    g = REPO / "docs/overnight/data/models_grader_gptoss.json"  # the gpt-oss sub-agent's derived numbers
    if g.exists():
        G = json.load(open(g))
        out["rates"]["gptoss_local"] = {
            "label": LABEL["gptoss_local"], "sub": f"{G['screen']['known_ge_0.9']} items", "local": True,
            "source": "docs/overnight/data/models_grader_gptoss.json",
            "arms": {a: {"v": v["p_B"][0], "lo": v["p_B"][1], "hi": v["p_B"][2], "n": v["runs"], "items": v["items"],
                         "n_B": v["n_B"]} for a, v in G["arms"].items()}}
    out["figure_order"] = [[sec, [m for m in ms if m in out["rates"]]] for sec, ms in ORDER]
    jc = defaultdict(Counter)
    if (D / "judge.jsonl").exists():
        for line in open(D / "judge.jsonl"):
            j = json.loads(line)
            jc[f"{j['model']}/{j['arm']}"][j["category"]] += 1
    out["judge_counts"] = {k: dict(v) for k, v in sorted(jc.items())}
    if (D / "by_eye.json").exists():
        out["by_eye"] = json.load(open(D / "by_eye.json"))
    cost = Counter()
    for line in open(D / "cost_log.jsonl"):
        r = json.loads(line)
        cost[r["model"]] += r["cost"]
    out["cost"] = {"by_model_usd": {k: round(v, 4) for k, v in cost.most_common()},
                   "total_usd": round(sum(cost.values()), 4), "cap_usd": CFG["cost_cap_usd"]}
    for p in sorted((D / "local").glob("*_lens.json")):
        out["lens"][p.stem] = json.load(open(p))
    dst = REPO / "docs/overnight/data/models_grader.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst} ({dst.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
