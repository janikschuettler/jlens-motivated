"""Collect the eval-awareness replication's derived numbers into
docs/experiments/evalaware-replication/data/evalaware_replication.json (plain python3; inputs: the analysis
JSONs copied from octavia to data/evalaware/).

  python3 scripts/explore/evalaware_build_data.py
"""

import json
from pathlib import Path

D = Path("data/evalaware")
OUT = Path("docs/experiments/evalaware-replication/data/evalaware_replication.json")


def r3(x):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, list):
        return [r3(v) for v in x]
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    return x


def pick(d, keys):
    return {k: r3(d[k]) for k in keys if k in d}


def main():
    c = json.loads((D / "analysis_corpus.json").read_text())
    t = json.loads((D / "analysis_top20.json").read_text())
    out = {"source": "docs/experiments/evalaware-replication/report.md",
           "paper": "Gurnee, Sofroniew et al. 2026, Verbalizable Representations Form a Global Workspace in Language "
                    "Models, transformer-circuits.pub/2026/workspace",
           "model": "Qwen/Qwen3-8B b968826d, thinking, J-lens neuronpedia/jacobian-lens qwen3-8b; workspace blocks 14-32 (even)",
           "score": "mean lens log-prob of ' fictional' and ' simulation' minus mean over 100 control nouns, "
                    "averaged over workspace layers"}
    out["step1a_top20"] = {k: {lens: pick(v[lens], ["positions", "eval_hits", "ctrl_hits", "eval_rate", "ctrl_rate"])
                               | {"top_tokens": dict(list(v[lens]["eval_tokens_seen"].items())[:8])}
                               for lens in ("jlens", "logit")} for k, v in t.items()}
    for lens in ("jlens", "logit"):
        r = c[lens]
        out[f"step1_{lens}"] = {
            "suite_means_last": r3(r["suite_means_last"]),
            "suite_means_cot": {k: r3(v["mean"]) for k, v in r["suite_means_cot"].items()},
            "E1b_auroc_eval_vs_deploy_last": r3(r["E1b_auroc_last"]),
            "E1b_n_eval_suites_above_both_deploy": r["E1b_n_eval_above_both_deploy"],
            "E1c_auroc_says_test_vs_not": r3(r["E1c_auroc_aware_vs_not"]),
            "E1c_within_suite": r3(r["E1c_within_suite"]),
            "E2b_auroc_eval_vs_deploy_cot": r3(r["E2b_auroc_cot"]),
            "paired": {k[len("paired_"):]: r3(v) for k, v in r.items() if k.startswith("paired_")},
            "suite_auroc_vs_deploy_last": r3(r["suite_auroc_vs_deploy_last"]),
        }
    out["says_test_by_suite"] = {k: pick(v, ["runs", "runs_aware", "prompts", "prompts_aware"])
                                 for k, v in c["aware_by_suite"].items()}
    out["blackmail_vllm"] = c["blackmail_vllm"]
    out["last_prompt_spread_between_samples_jlens"] = r3(c["last_prompt_spread_between_samples_jlens"])
    for name, f in (("grader_qwen3_8b_nothink", "analysis_grader_nt8.json"),
                    ("grader_qwen3_14b_think", "analysis_grader_q14.json"),
                    ("hack_warmstart_hidden_vs_nohack", "analysis_hack.json")):
        d = list(json.loads((D / f).read_text()).values())[0]
        out[name] = {"counts": d["counts"]} | {k: r3({kk: v[kk] for kk in ("auroc", "ci", "n_items", "n_pos", "n_neg",
                                                                               "mean_pos", "mean_neg",
                                                                               "mean_allowed_ref") if kk in v})
                                               if isinstance(v, dict) else r3(v)
                                               for k, v in d.items() if k != "counts"}
    for f, key in (("analysis_ablate.json", "ablation"), ("analysis_steer.json", "steering")):
        if (D / f).exists():
            out[key] = r3(json.loads((D / f).read_text()))
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
