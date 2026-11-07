"""C3-R stage 0 (stdlib): fresh known items from the screen (acc.A >= 0.9, not in known_items.jsonl), and after the
thinking-off screen, merge its counts as n_correct.B / n.B.
  python3 scripts/explore/c3r_stage0.py items      # needs data/explore/c3r/known_all.jsonl (w2_known.py output)
  python3 scripts/explore/c3r_stage0.py merge      # needs data/explore/c3r/nothink_known.jsonl (w2_known.py --setting B)
"""
import json, sys
from pathlib import Path
rows = lambda p: [json.loads(l) for l in open(p) if l.strip()]
D = Path("data/explore/c3r")
if sys.argv[1] == "items":
    old = {d["id"] for d in rows("data/explore/known_items.jsonl")}
    allr = rows(D / "known_all.jsonl")
    fresh = [d for d in allr if d["id"] not in old and d["acc"]["A"] >= 0.9]
    assert all(d.get("hinted") for d in fresh), "no hinted letter"
    (D / "items.jsonl").write_text("".join(json.dumps(d) + "\n" for d in fresh))
    print(f"screened {len(allr)}, in old pool {sum(d['id'] in old for d in allr)}, known fresh {len(fresh)}")
else:
    nt = {d["id"]: d for d in rows(D / "nothink_known.jsonl")}
    out = []
    for d in rows(D / "items.jsonl"):
        e = nt[d["id"]]
        d["n_correct"]["B"], d["n"]["B"], d["acc"]["B"] = e["n_correct"]["B"], e["n"]["B"], e["acc"]["B"]
        out.append(d)
    (D / "items_with_B.jsonl").write_text("".join(json.dumps(d) + "\n" for d in out))
    print(len(out), "items with B counts")
