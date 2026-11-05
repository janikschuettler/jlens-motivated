"""Figure for subtle-cues-cross-model: switch rate with 95% CI per model x cue. Stdlib only; reads data/scm.json."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[2]; X = R / "docs/experiments/subtle-cues-cross-model"
d = json.load(open(X / "data/scm.json"))
models = ["gemma-3-27b-it", "llama-3.3-70b-it", "gpt-oss-20b", "deepseek-v4-flash", "qwen3-32b", "qwen3-14b"]
cues = [("W_answer", "states the answer", "#444"), ("metadata", "metadata tag", "#1f77b4"), ("grader_code", "grader code", "#e08a00"), ("fewshot_marker", "marked examples", "#2a9d57"), ("fewshot_always_b", "always-B examples", "#9a4fb0")]
W, rowh, x0, xs = 640, 62, 150, 4.4
H = 70 + rowh * len(models)
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11"><rect width="100%" height="100%" fill="white"/>']
o.append('<text x="10" y="18" font-size="13" fill="#222">Share of runs ending on the hinted wrong option (95% CI over items)</text>')
for i, (k, n, c) in enumerate(cues):
    o.append(f'<circle cx="{150+i*95}" cy="38" r="4" fill="{c}"/><text x="{157+i*95}" y="42" fill="#222" font-size="10">{n}</text>')
for p in range(0, 101, 20):
    x = x0 + p * xs; o.append(f'<line x1="{x}" x2="{x}" y1="52" y2="{H-14}" stroke="#ddd"/><text x="{x}" y="{H-2}" text-anchor="middle" fill="#555">{p}%</text>')
for r, m in enumerate(models):
    y = 60 + r * rowh; o.append(f'<text x="10" y="{y+rowh/2-4}" fill="#222">{m}</text>')
    for i, (k, n, c) in enumerate(cues):
        e = d[m][k]; yy = y + 8 + i * 10
        o.append(f'<line x1="{x0+e["ci"][0]*100*xs}" x2="{x0+e["ci"][1]*100*xs}" y1="{yy}" y2="{yy}" stroke="{c}" stroke-width="2"/><circle cx="{x0+e["switch"]*100*xs}" cy="{yy}" r="3.5" fill="{c}"/>')
o.append('</svg>')
(X / "fig_scm_switch.svg").write_text("\n".join(o))
