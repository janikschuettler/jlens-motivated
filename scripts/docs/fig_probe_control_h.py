"""Figure for probe-control PC-H: within-prompt AUROC (dots, 95% bars) on the test-awareness runs."""
import json
d = json.load(open("docs/experiments/probe-control/data/pc_test_a_second_dataset.json"))
rows = [("Nguyen probe", "nguyen"), ("Hua direction", "hua"), ("J-lens eval words", "jlens"), ("Logit-lens eval words", "logit"), ("Window length", "length"), ("Word n-gram text", "text")]
labs = [("keyword", "Keyword label (185 prompts)", "#2a6f97"), ("strict_yes_vs_no", "Strict judge label (51 prompts)", "#c1440e")]
W, H, x0, x1, y0 = 760, 380, 230, 720, 70
X = lambda v: x0 + (v - 0.3) / 0.7 * (x1 - x0)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">', f'<rect width="{W}" height="{H}" fill="#fff"/>',
     '<text x="20" y="24" font-size="14" font-weight="bold">Same prompt, runs that will say "testing me" vs those that will not (Qwen3-14B test-awareness runs)</text>',
     '<text x="20" y="42" fill="#555">Within-prompt AUROC, 0.5 = chance; dots with 95% intervals (resampling prompts). Grey band: 100 random directions (Nguyen layer).</text>']
for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    s.append(f'<line x1="{X(t)}" x2="{X(t)}" y1="{y0-8}" y2="{y0+len(rows)*48}" stroke="{"#444" if t==0.5 else "#ddd"}"/><text x="{X(t)-9}" y="{y0+len(rows)*48+16}" fill="#555">{t:.1f}</text>')
b = d["keyword"]["random_band"]["nguyen_layer"]
s.append(f'<rect x="{X(b["q025"])}" y="{y0-8}" width="{X(b["q975"])-X(b["q025"])}" height="{len(rows)*48}" fill="#999" opacity="0.18"/>')
for i, (nm, key) in enumerate(rows):
    y = y0 + i * 48 + 14
    s.append(f'<text x="20" y="{y+14}">{nm}</text>')
    for j, (lk, ln, col) in enumerate(labs):
        r = d[lk]["primary"][key]; yy = y + j * 16
        s.append(f'<line x1="{X(r["ci95"][0])}" x2="{X(r["ci95"][1])}" y1="{yy}" y2="{yy}" stroke="{col}" stroke-width="2"/><circle cx="{X(r["auroc"])}" cy="{yy}" r="4.5" fill="{col}"/>')
for j, (lk, ln, col) in enumerate(labs):
    s.append(f'<circle cx="{x0+j*250}" cy="{H-24}" r="4.5" fill="{col}"/><text x="{x0+j*250+10}" y="{H-20}">{ln}</text>')
s.append("</svg>")
open("docs/experiments/probe-control/fig_probe_control_h.svg", "w").write("\n".join(s))
