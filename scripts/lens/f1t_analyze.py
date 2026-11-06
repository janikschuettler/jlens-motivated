"""F1-T (docs/preregistration.md "F1-T"): pooled old+new equivalence tests for the topped-up cells.
Same machinery as firming_f1_cases.py / firming_f1_scores.py (5,000 item bootstraps, seed 0, +-0.10).

    python scripts/lens/f1t_analyze.py --old data/f1t/orig --new data/f1t/new --t2-old data/f1t/orig/t2.jsonl \
        --t2-new data/f1t/new/t2_new.jsonl --out docs/experiments/firming-f1-f2/data/f1t_results.json
The new dir holds wording.npz / c3.npz from competitors_extract.py (one or more stages concatenated by --new nargs).
"""
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import firming_f1_cases as fc
import firming_f1_scores as fs


def concat_dir(dirs, name, out):
    lens, runs, meta = [], [], None
    for d in dirs:
        p = Path(d) / f"{name}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        m = json.loads(str(z["meta_json"])); runs += m["runs"]; lens.append(z["lens"]); meta = m
    if not lens:
        return False
    meta = dict(meta); meta["runs"] = runs
    np.savez(Path(out) / f"{name}.npz", lens=np.concatenate(lens), meta_json=json.dumps(meta))
    return True


def verdict(c):
    j = c["jlens"]
    if j["equiv_chance"]:
        return "equivalent to chance"
    if j["differs_from_chance"]:
        return "signal (95% interval excludes 0.5)"
    return "inconclusive"


def case_verdict(cells):
    if all(c["jlens"]["equiv_chance"] for c in cells.values()):
        return "equivalent to chance (all positions)"
    if any(c["jlens"]["differs_from_chance"] for c in cells.values()):
        return "signal at some position"
    return "inconclusive"


def diff_verdict(c):
    d = c["diff"]
    if d["ci95"][0] > 0:
        return "J ahead (95% interval above 0)"
    if d["ci95"][1] < 0:
        return "logit ahead (95% interval below 0)"
    if d["equivalent_lenses"]:
        return "lenses equivalent (within 0.10)"
    if d["advantage_ge_010_ruled_out"]:
        return "J advantage >= 0.10 ruled out"
    return "J-logit inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True); ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--t2-old"); ap.add_argument("--t2-new", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for case in ("wording", "c3"):
        res[case] = {}
        for tag, dirs in (("old", [a.old]), ("new", a.new), ("pooled", [a.old] + a.new)):
            with tempfile.TemporaryDirectory() as td:
                if not concat_dir(dirs, case, td):
                    continue
                r = fc.run_case(case, td, np.random.default_rng(fc.SEED))
            r["case_verdict_jlens"] = case_verdict(r["cells"])
            for pos, c in r["cells"].items():
                c["verdict_jlens"] = verdict(c); c["verdict_diff"] = diff_verdict(c)
            res[case][tag] = r
            print(case, tag, r["n_pos"], r["n_neg"], r["n_items"], r["case_verdict_jlens"],
                  {p: (round(c["jlens"]["point"], 3), [round(x, 3) for x in c["jlens"]["ci90"]], round(c["diff"]["point"], 3),
                       [round(x, 3) for x in c["diff"]["ci90"]]) for p, c in r["cells"].items()}, flush=True)
    if a.t2_old:
        def load(p):
            return [{"item": d["item"], "pos": int(d["pos"]), "jlens": d["jlens"], "logit": d["logit"]}
                    for d in map(json.loads, open(p)) if "run_id" in d]
        old = load(a.t2_old); new = [r for p in a.t2_new for r in load(p)]
        res["t2"] = {}
        for tag, rows in (("old", old), ("new", new), ("pooled", old + new)):
            if not rows:
                continue
            r = fs.run(rows)
            r["verdict_jlens"] = verdict({"jlens": r["jlens"]}); r["verdict_diff"] = diff_verdict({"diff": r["diff"]})
            res["t2"][tag] = r
            print("t2", tag, r["n_pos"], r["n_neg"], r["n_clusters"], r["verdict_jlens"], round(r["jlens"]["point"], 3),
                  [round(x, 3) for x in r["jlens"]["ci90"]], r["verdict_diff"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
