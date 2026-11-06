"""F1-T figure (stdlib SVG): J-lens AUROC with 90% interval per cell and position, old / new / pooled, against the
equivalence band 0.40-0.60; right panel: J minus logit lens (90% interval) against +-0.10.
    python scripts/lens/f1t_figure.py RESULTS.json OUT.svg
"""
import json, sys
res = json.load(open(sys.argv[1]))
rows = []  # (label, tag, point, lo, hi, dpoint, dlo, dhi)
def add(label, tag, c):
    rows.append((label, tag, c["jlens"]["point"], *c["jlens"]["ci90"], c["diff"]["point"], *c["diff"]["ci90"]))
names = {"wording": "Hint followers vs resisters", "c3": "Planted claim kept vs recovered"}
pos = {"last_prompt": "last prompt token", "mid": "mid reasoning", "think_end": "end of reasoning", "answer": "answer"}
for case in ("wording", "c3"):
    for p in ("last_prompt", "mid", "think_end"):
        for tag in ("old", "new", "pooled"):
            if tag in res.get(case, {}) and p in res[case][tag]["cells"]:
                add(f"{names[case]}, {pos[p]}", tag, res[case][tag]["cells"][p])
for tag in ("old", "new", "pooled"):
    if tag in res.get("t2", {}):
        add("Followers who still know vs persuaded, answer", tag, res["t2"][tag])
W, rowh, left, pw = 980, 15, 330, 260
H = 70 + rowh * len(rows) + 20 * (len(rows) // 3)
col = {"old": "#8a8f98", "new": "#2a7de1", "pooled": "#d9480f"}
def x1(v): return left + (min(max(v, 0.2), 0.8) - 0.2) / 0.6 * pw
def x2(v): return left + pw + 60 + (min(max(v, -0.3), 0.3) + 0.3) / 0.6 * pw
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">',
     '<rect width="100%" height="100%" fill="white"/>',
     f'<text x="10" y="18" font-size="13" font-weight="bold">F1-T top-up: J-lens vs chance (left) and J-lens minus logit lens (right), 90% intervals</text>',
     f'<rect x="{x1(.4)}" y="30" width="{x1(.6)-x1(.4)}" height="{H-50}" fill="#e8f3e8"/>',
     f'<rect x="{x2(-.1)}" y="30" width="{x2(.1)-x2(-.1)}" height="{H-50}" fill="#e8f3e8"/>',
     f'<line x1="{x1(.5)}" x2="{x1(.5)}" y1="30" y2="{H-20}" stroke="#555"/>', f'<line x1="{x2(0)}" x2="{x2(0)}" y1="30" y2="{H-20}" stroke="#555"/>']
for v in (.2, .4, .5, .6, .8):
    o.append(f'<text x="{x1(v)}" y="{H-6}" text-anchor="middle">{v}</text>')
for v in (-.3, -.1, 0, .1, .3):
    o.append(f'<text x="{x2(v)}" y="{H-6}" text-anchor="middle">{v:+.1f}</text>')
y = 42; last = None
for lab, tag, pt, lo, hi, dp, dlo, dhi in rows:
    if lab != last:
        y += 6; o.append(f'<text x="8" y="{y+10}">{lab}</text>'); last = lab
    c = col[tag]
    o.append(f'<line x1="{x1(lo)}" x2="{x1(hi)}" y1="{y+7}" y2="{y+7}" stroke="{c}" stroke-width="2"/><circle cx="{x1(pt)}" cy="{y+7}" r="3" fill="{c}"/>')
    o.append(f'<line x1="{x2(dlo)}" x2="{x2(dhi)}" y1="{y+7}" y2="{y+7}" stroke="{c}" stroke-width="2"/><circle cx="{x2(dp)}" cy="{y+7}" r="3" fill="{c}"/>')
    y += rowh
for i, (t, c) in enumerate(col.items()):
    o.append(f'<circle cx="{left+i*90}" cy="{H-34}" r="4" fill="{c}"/><text x="{left+i*90+8}" y="{H-30}">{t}</text>')
o.append('</svg>')
open(sys.argv[2], "w").write("\n".join(o))
