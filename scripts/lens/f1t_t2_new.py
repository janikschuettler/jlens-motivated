"""F1-T: per-run T2 scores on the new wording runs (numpy only). Same as firming_recompute.do_t2: runs of the four T2
arms with final answer = pushed letter, on known items, full-history sincere re-asking (fork) names the correct
letter (positive) or the pushed letter (negative), other letters excluded; S = lp(correct) - mean lp(the two
non-answer wrong letters) at the pre_answer position, mean over layers 14-32; J-lens and logit lens.

    python scripts/lens/f1t_t2_new.py --readouts DIR --forks forks.jsonl --known items.jsonl [items2.jsonl] --out t2_new.jsonl
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import firming_recompute as fr

ap = argparse.ArgumentParser()
ap.add_argument("--readouts", required=True); ap.add_argument("--forks", nargs="+", required=True)
ap.add_argument("--known", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
rdir = Path(a.readouts)
known = {d["id"] for f in a.known for d in fr.jl(f) if (d.get("acc") or {}).get("A", 0) >= 0.9}
reel, pl = fr.reel_map(a.forks, "full")
rows, seen, n_other, n_noreel, layers_seen = [], set(), 0, 0, set()
for m in fr.jl(rdir / "index.jsonl"):
    if m["run_id"] in seen:
        continue
    seen.add(m["run_id"])
    ex = fr.parse(m.get("extras")) if isinstance(m.get("extras"), str) else m.get("extras") or {}
    arm = ex.get("arm") if isinstance(ex, dict) else None
    q, C, A, H = m["question_id"], m["correct_letter"], m["answer"], m["hint_letter"]
    if arm not in fr.T2_ARMS or q not in known or H is None or A != H:
        continue
    if m["named_positions"].get("pre_answer", -1) < 0:
        continue
    r = reel.get(m["run_id"])
    if r is None:
        n_noreel += 1; continue
    if r not in (C, H):
        n_other += 1; continue
    others = [x for x in m["letters"] if x not in (C, A)]
    S, layers, mass = fr.margin_at(rdir / m["file"], m, m["named_positions"]["pre_answer"], others)
    layers_seen.add(tuple(layers))
    rows.append({"run_id": m["run_id"], "item": q, "arm": arm, "pos": r == C, "reel": r, "jlens": float(S[0]),
                 "logit": float(S[1]), "mass_written": mass})
with open(a.out, "w") as f:
    f.write(json.dumps({"summary": {"n_rows": len(rows), "excluded_other_letter": n_other,
                                    "followers_without_reelicitation": n_noreel, "layers": sorted(layers_seen)}}) + "\n")
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "rows", sum(r["pos"] for r in rows), "positive; other-letter", n_other, "no reelicit", n_noreel)
