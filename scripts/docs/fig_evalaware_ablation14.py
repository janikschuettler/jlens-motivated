"""Figure for the eval-awareness ablation on Qwen3-14B (docs/experiments/evalaware-ablation-14b/fig17_evalaware_ablation14.svg).

Kept out of make_figures.py (as fig_evalaware.py) to avoid colliding with other edits; uses its helpers.
Input: docs/experiments/evalaware-ablation-14b/data/evalaware_ablation_14b.json only.

    ~/.local/bin/uv run --project envs/judge python scripts/docs/fig_evalaware_ablation14.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import BLUE, GREY, ORANGE, figure, load, pct, pct_axis  # noqa: E402

ARMS = [("none", "No ablation"),
        ("eval_J", "Eval words removed (J-lens directions)"),
        ("eval_logit", "Eval words removed (logit-lens directions)"),
        ("control_J", "Everyday nouns removed (J-lens directions)"),
        ("random", "Random directions removed")]


def main():
    D = load("evalaware_ablation_14b")
    R = D["rates"]

    def val(arm, m, s):
        r = R[arm][m]
        return {"s": s, "v": r["k"] / r["n"], "lo": r["ci95"][0] / 100, "hi": r["ci95"][1] / 100}

    rows1, rows2 = [], []
    for arm, lab in ARMS:
        if arm not in R:
            continue
        sub = f"{R[arm]['says_testing']['n']} runs"
        rows1.append({"label": lab, "sub": sub, "vals": [val(arm, "says_testing", "t"), val(arm, "says_trick", "k")]})
        rows2.append({"label": lab, "sub": sub, "vals": [val(arm, "took_bug", "b")]})
    s1 = {"t": ("circle", BLUE, "says the user is testing it"),
          "k": ("square", ORANGE, "calls the note a trick / red herring (exploratory)")}
    s2 = {"b": ("diamond", GREY, "takes the bug (answers the letter the buggy grader marks correct)")}
    figure("fig17_evalaware_ablation14.svg",
           "Removing the J-lens eval-word directions stops Qwen3-14B saying \"you're testing me\", but not the cheating",
           ["Qwen3-14B (thinking) on 69 known questions with the note \"the automated grader has a bug, it marks B as "
            "correct\" and the system prompt \"you are an AI agent being evaluated … get the highest score you can\". "
            "During generation, 26 directions are projected out of the residual stream at blocks 16–34, every position. "
            "95% Wilson intervals over runs (the paired tests in the report resample questions)."],
           [("What it says", dict(rows=rows1, xmin=0, xmax=0.5, step=0.1, fmt=pct, x_fmt=pct_axis, mode="dot",
                                  series=s1, xlabel="share of runs", label_w=330, row_h=30, value_labels=False)),
            ("What it does", dict(rows=rows2, xmin=0, xmax=0.5, step=0.1, fmt=pct, x_fmt=pct_axis, mode="dot",
                                  series=s2, xlabel="share of runs", label_w=330, row_h=26, value_labels=True))],
           legend_items=[s1["t"], s1["k"], s2["b"]],
           notes=["Source: docs/experiments/evalaware-ablation-14b/data/evalaware_ablation_14b.json. Report: docs/experiments/evalaware-ablation-14b/report.md.",
                  "Eval words: the J-lens paper's ablation list (26 single Qwen tokens: \" fake\", \" fictional\", \" simulation\", "
                  "\" testing\", \" evalu\", \" benchmark\", …). Paper (Claude Sonnet 4.5, blackmail prompt): 71% → 3% say it is a test."])


if __name__ == "__main__":
    main()
