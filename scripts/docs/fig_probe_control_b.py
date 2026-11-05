"""Figure for probe-control test (b) from docs/experiments/probe-control/data/pc_test_b.json (stdlib only)."""
import json
p = "docs/experiments/probe-control/data/pc_test_b.json"
o = json.load(open(p))
names = [("nguyen", "Nguyen probe"), ("hua", "Hua direction"), ("jlens", "J-lens eval-word score"), ("logit", "Logit-lens score"), ("text", "Text n-grams (zero-shot)"), ]
rows = []
for k, lab in names:
    for key, tag in (("primary_matched_P1-P4", "same stated belief, P1-P4"), ("P1_matched", "format user to benchmark"), ("P3_matched", "eval words in grader prompt"), ("P5_matched", "AI-topic sentence (control)")):
        v = o["pair_auroc"][k][key]
        if v: rows.append((f"{lab}, {tag}", v, key.startswith("primary")))
W, H = 900, 70 + 22 * len(rows)
x = lambda a: 400 + a * 440
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="13"><rect width="{W}" height="{H}" fill="white"/>',
     '<text x="10" y="22" font-size="15" font-weight="bold">Does the score rise when only the surface changes? Pair AUROC (95% CI)</text>',
     f'<line x1="{x(0.5)}" y1="35" x2="{x(0.5)}" y2="{H-22}" stroke="#888" stroke-dasharray="4"/><text x="{x(0.5)-28}" y="{H-6}" fill="#555">0.5 = no change</text>']
for i, (n, v, prim) in enumerate(rows):
    y = 55 + i * 22; c = "#2b6cb0" if prim else "#999"
    s.append(f'<text x="10" y="{y+4}">{n} (n={v["n_pairs"]})</text><line x1="{x(v["ci95"][0])}" y1="{y}" x2="{x(v["ci95"][1])}" y2="{y}" stroke="{c}" stroke-width="3"/><circle cx="{x(v["auroc"])}" cy="{y}" r="5" fill="{c}"/><text x="{x(v["ci95"][1])+8}" y="{y+4}">{v["auroc"]}</text>')
s.append("</svg>")
open("docs/experiments/probe-control/fig_probe_control_b.svg", "w").write("\n".join(s))
