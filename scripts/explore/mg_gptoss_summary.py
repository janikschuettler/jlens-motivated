"""Derived numbers for gpt-oss-20b Phase 2 -> docs/overnight/data/models_grader_gptoss.json (CPU).

  python scripts/explore/mg_gptoss_summary.py --out docs/overnight/data/models_grader_gptoss.json

Inputs (octavia data/): explore/models_grader/gptoss_screen/labels.jsonl, gptoss_known_items.jsonl,
gptoss_arms/labels.jsonl, gptoss_arms/follower_kw_counts.json, m0/*gpt-oss-20b.json,
readouts/mg_gptoss_analysis.json, readouts/mg_gptoss_m04.json. Item-bootstrap 95% intervals.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

D = Path("data")


def boot(by_item, n=2000, seed=0):
    keys = sorted(by_item)
    rng = random.Random(seed)
    means = {k: sum(v) / len(v) for k, v in by_item.items()}
    point = sum(sum(v) for v in by_item.values()) / sum(len(v) for v in by_item.values())
    st = sorted(sum(means[rng.choice(keys)] for _ in keys) / len(keys) for _ in range(n))
    return [round(point, 4), round(st[int(0.025 * n)], 4), round(st[int(0.975 * n) - 1], 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"model": "openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee",
           "precision": "MXFP4 experts (native release format), bf16 elsewhere; vLLM 0.33 MARLIN MXFP4 MoE for "
                        "generation; transformers 5.11 + kernels 0.12.3 (Mxfp4GptOssExperts, eager attention) for readouts",
           "lens": "neuronpedia/jacobian-lens@e166deb5 gpt-oss-20b/jlens/Salesforce-wikitext (blocks 0-22, fp16; "
                   "fit run dtype bfloat16, dequantisation not recorded)",
           "sampling": {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 6144, "reasoning_effort": "medium"},
           "workspace_layers": [9, 11, 13, 15, 17, 19, 21]}
    scr = defaultdict(list)
    for l in open(D / "explore/models_grader/gptoss_screen/labels.jsonl"):
        r = json.loads(l)
        scr[r["question_id"]].append(r["answer"] == r["correct_letter"])
    out["screen"] = {"items": len(scr), "samples_per_item": 10, "runs": sum(len(v) for v in scr.values()),
                     "acc": boot({k: [float(x) for x in v] for k, v in scr.items()}),
                     "known_ge_0.9": sum(sum(v) / len(v) >= 0.9 for v in scr.values())}
    arms = defaultdict(lambda: defaultdict(list))
    none = defaultdict(int)
    for l in open(D / "explore/models_grader/gptoss_arms/labels.jsonl"):
        r = json.loads(l)
        arms[r["arm"]][r["question_id"]].append(r)
    out["arms"] = {}
    for arm, per in arms.items():
        rows = [r for v in per.values() for r in v]
        out["arms"][arm] = {
            "runs": len(rows), "items": len(per),
            "p_B": boot({k: [float(r["answer"] == r["letter"]) for r in v] for k, v in per.items()}),
            "p_correct": boot({k: [float(r["answer"] == r["correct_letter"]) for r in v] for k, v in per.items()}),
            "n_B": sum(r["answer"] == r["letter"] for r in rows),
            "n_unanswered": sum(r["answer"] is None for r in rows),
            "n_truncated": sum(r["finish_reason"] == "length" for r in rows),
            "mean_completion_tokens": round(sum(r["completion_tokens"] for r in rows) / len(rows), 1),
        }
    out["follower_keyword_labels"] = json.loads((D / "explore/models_grader/gptoss_arms/follower_kw_counts.json").read_text())
    out["m0"] = {}
    for f in sorted((D / "m0").glob("*gpt-oss-20b.json")):
        d = json.loads(f.read_text())
        out["m0"][f.stem] = {"all_passed": d["all_passed"], "checks": [(c["check"], c["passed"], c["detail"]) for c in d["checks"]]}
    out["m04_all_readout_runs"] = json.loads((D / "readouts/mg_gptoss_m04.json").read_text())
    an = json.loads((D / "readouts/mg_gptoss_analysis.json").read_text())
    keep = ("corr@pre_answer", "sal@pre_answer", "corr@think_end", "sal@think_end",
            "concept_grader@last_prompt", "concept_grader@cot_mean", "concept_grader@pre_answer",
            "concept_truth@cot_mean", "concept_truth@pre_answer")
    out["lens_groups"] = {g: {"n": v["n"], **{k: v[k] for k in keep if k in v}} for g, v in an["groups"].items()}
    out["lens_contrasts"] = an["contrasts"]
    out["lens_layers"] = an["layers"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("screen", "arms")}, indent=1))


if __name__ == "__main__":
    main()
