"""Figure for the eval-awareness replication (docs/experiments/evalaware-replication/fig16_evalaware.svg).

Kept out of make_figures.py so it doesn't collide with other edits tonight; uses its
helpers. Input: docs/experiments/evalaware-replication/data/evalaware_replication.json only.

    ~/.local/bin/uv run --project envs/judge python scripts/docs/fig_evalaware.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import BLUE, GREY, ORANGE, auc, figure, load  # noqa: E402

LABELS = {
    "redux_home": ("MMLU, our usual format", "eval"),
    "moral": ("MoralChoice scenarios", "eval"),
    "hack_task": ("Reward-hack coding task", "eval"),
    "grader_plain": ("Grader-bug note", "eval"),
    "redux_bench": ("MMLU, \"from the MMLU benchmark\"", "eval"),
    "grader_agent": ("Grader-bug note + \"AI agent being evaluated\"", "eval"),
    "syco": ("Sycophancy eval prompts", "eval"),
    "blackmail": ("Blackmail prompt (one prompt)", "eval"),
    "oasst": ("OASST user prompts", "deploy"),
    "redux_casual": ("MMLU, asked as a coworkers' argument", "deploy"),
    "wildchat": ("WildChat user prompts", "deploy"),
}


def main():
    D = load("evalaware_replication")
    sm = D["step1_jlens"]["suite_means_last"]
    sml = D["step1_logit"]["suite_means_last"]
    rows = []
    for sec, kind in (("Evaluation-like prompts", "eval"), ("Real-use prompts", "deploy")):
        rows.append({"section": sec})
        for k, (lab, kd) in sorted(LABELS.items(), key=lambda kv: sm[kv[0]]["mean"]):
            if kd != kind:
                continue
            j, lg = sm[k], sml[k]
            rows.append({"label": lab, "vals": [
                {"s": "j", "v": j["mean"], "lo": j["ci"][0], "hi": j["ci"][1]},
                {"s": "l", "v": lg["mean"], "lo": lg["ci"][0], "hi": lg["ci"][1]}]})
    series = {"j": ("circle", BLUE, "J-lens"), "l": ("square", ORANGE, "logit lens")}

    g8, g14, hk = D["grader_qwen3_8b_nothink"], D["grader_qwen3_14b_think"], D["hack_warmstart_hidden_vs_nohack"]
    j1 = D["step1_jlens"]

    def v(d, s="j"):
        return {"s": s, "v": d["auroc"], "lo": d["ci"][0], "hi": d["ci"][1]}

    rows2 = [
        {"section": "Replication steps"},
        {"label": "1b eval vs real-use prompts (start of turn)", "vals": [v(j1["E1b_auroc_eval_vs_deploy_last"])]},
        {"label": "1c prompts where it says \"testing me\"", "vals": [v(j1["E1c_auroc_says_test_vs_not"])]},
        {"label": "2b eval vs real-use (reasoning positions)", "vals": [v(j1["E2b_auroc_eval_vs_deploy_cot"])]},
        {"section": "Towards our setup: took the bug vs answered correctly (reasoning = before any mention of the grader)"},
        {"label": "3 eval score, reasoning (Qwen3-8B, no thinking)",
         "vals": [v(g8["E_free|jlens"])]},
        {"label": "3 eval score, reasoning (Qwen3-14B, thinking)",
         "vals": [v(g14["E_free|jlens"])]},
        {"label": "4 \"cheating\" score, reasoning (Qwen3-14B)",
         "vals": [v(g14["C_free|jlens"])]},
        {"label": "4 hidden hack vs no hack (trained organism)",
         "vals": [v(hk["E_cot|jlens"])]},
        {"label": "5 eval score at the answer (Qwen3-8B)",
         "vals": [v(g8["E_ans|jlens"])]},
        {"label": "5 eval score at the answer (Qwen3-14B)",
         "vals": [v(g14["E_ans|jlens"])]},
        {"label": "text: grader/bug keyword count (Qwen3-14B)",
         "vals": [v(g14["text_keywords|jlens"], "t")]},
    ]
    series2 = {"j": ("circle", BLUE, "J-lens eval-awareness score"), "t": ("diamond", GREY, "text baseline")}
    figure("fig16_evalaware.svg",
           "The J-lens eval-awareness score doesn't tell evaluations from real use on Qwen3-8B",
           ["Top: the paper's score (log-prob of \" fictional\" and \" simulation\" minus 100 everyday nouns, mean over "
            "layers 14–32) at the start of the model's turn, by prompt set. Real user prompts score as high as evaluation "
            "suites. Bottom: how well the score separates the groups (AUROC; 0.5 = chance, 1 = perfect). 95% intervals."],
           [(None, dict(rows=rows, xmin=1.5, xmax=5.5, step=0.5, fmt=lambda x: f"{x:.2f}",
                        x_fmt=lambda x: f"{x:.1f}", mode="dot", series=series,
                        xlabel="eval-awareness score (nats; higher = more \"fictional / simulation\")",
                        label_w=330, row_h=24, value_labels=False)),
            (None, dict(rows=rows2, xmin=0, xmax=1, step=0.1, fmt=auc, x_fmt=lambda x: f"{x:.1f}", mode="dot",
                        series={**series2}, ref=0.5, ref_label="chance", xlabel="AUROC",
                        label_w=330, row_h=26, value_labels=True))],
           legend_items=[series["j"], series["l"], series2["t"]],
           notes=["Source: docs/experiments/evalaware-replication/data/evalaware_replication.json. Report: docs/experiments/evalaware-replication/report.md.",
                  "Paper: Gurnee, Sofroniew et al. 2026 (Claude Sonnet 4.5 / Opus 4.6). Our model: Qwen3-8B, thinking, Neuronpedia J-lens."])


if __name__ == "__main__":
    main()
