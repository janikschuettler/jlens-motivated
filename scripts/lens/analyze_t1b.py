"""Pre-registered test T1b (docs/overnight/preregistration.md): T1 on fresh items.
Lens env, CPU. Reuses the T1 code in analyze_prereg.py (score S, verdict) and
analyze_main.auroc_boot (item bootstrap) unchanged.

Items: the T1b screens (setting B, item seed 3 draw, 10 unhinted samples each);
known = >= 9/10 correct. Runs: final answer = hinted B in the M&B-hint arm,
non-thinking. Groups: silent (labels.jsonl "mentions" empty) vs open.
Pass: J-lens AUROC(S; silent vs open) lower bound > 0.5; underpowered if < 15
items with a silent switch. Predicted: logit-lens interval includes 0.5.
Exploratory: per-layer AUROCs, AUROC within completion-length tertiles.

Two stages, so that readouts on two machines need not be copied:
  score: --part SCREEN_LABELS:HINTED_LABELS:READOUT_DIR[,DIR2][:ITEMS_ALLOWED_JSONL] (paths under
         --data) -> --scores-out, one JSON line per scored switch (S, per-layer margins,
         group, item, length) plus a header line with the part's counts.
  test:  --scores FILE [FILE ...] -> --out, the pre-registered test and exploratory parts.
Items are counted once: ITEMS_ALLOWED_JSONL restricts a part to the items listed there.

uv run --project envs/lens python scripts/lens/analyze_t1b.py score --scores-out data/readouts/t1b_oct_scores.jsonl \
  --part explore/t1b_oct/screen/labels.jsonl:explore/t1b_oct/hinted/labels.jsonl:readouts/t1b_oct
uv run --project envs/lens python scripts/lens/analyze_t1b.py test --out data/readouts/prereg_t1b.json \
  --scores data/readouts/t1b_oct_scores.jsonl data/readouts/t1b_mayor_scores.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_main import auroc_boot, boot_mean  # noqa: E402
from analyze_prereg import SEED, load_set, parse, score_run, verdict_t  # noqa: E402

LAYERS = list(range(14, 33, 2))


def score(args) -> None:
    D = Path(args.data)
    out = open(args.scores_out, "w")
    for part in args.part:
        f = part.split(":")
        screen_p, hinted_p, rdir_p = f[:3]
        allowed = {json.loads(line)["id"] for line in open(D / f[3])} if len(f) > 3 else None
        n_ok, n_all = Counter(), Counter()
        for d in map(json.loads, open(D / screen_p)):
            if allowed is not None and d["question_id"] not in allowed:
                continue
            n_all[d["question_id"]] += 1
            n_ok[d["question_id"]] += d["outcome"] == "correct"
        if set(n_all.values()) != {10} or (allowed is not None and set(n_all) != allowed):
            raise ValueError(f"{screen_p}: samples per item {Counter(n_all.values())}, {len(n_all)} items")
        known = {q for q in n_all if n_ok[q] >= 9}
        labels = {d["run_id"]: d for d in map(json.loads, open(D / hinted_p))}
        hinted_items = {d["question_id"] for d in labels.values()}
        if not known <= hinted_items:
            raise ValueError(f"{part}: {len(known - hinted_items)} known items have no hinted runs")
        sw = {r for r, d in labels.items() if d["outcome"] == "switch" and d["question_id"] in known}
        rows, no_anchor = [], []
        metas = [(D / rp, m) for rp in rdir_p.split(",") for m in load_set(D / rp)]  # READOUT_DIR may be "a,b"
        for rdir, m in metas:
            q, A, H = m["question_id"], m["answer"], m["hint_letter"]
            if q not in known or H is None or A != H or m["enable_thinking"]:
                continue
            lab = labels.get(m["run_id"])
            if lab is None:
                raise ValueError(f"no label for {m['run_id']}")
            if lab["answer"] != A:
                raise ValueError(f"{m['run_id']}: label answer {lab['answer']} != readout answer {A}")
            z = np.load(rdir / m["file"])
            sc = score_run(z, m)
            if sc is None:  # as in T1 (run_t1): runs without a pre_answer anchor are skipped, but counted
                no_anchor.append(m["run_id"])
                continue
            rows.append({"run_id": m["run_id"], "item": q, "part": rdir_p,
                         "group": "open" if parse(lab.get("mentions")) else "silent", "S": sc["S"].tolist(),
                         "layer_margin": sc["layer_margin"].tolist(), "len": lab["completion_tokens"]})
        if not args.partial and {r["run_id"] for r in rows} | set(no_anchor) != sw:
            raise ValueError(f"{part}: {len(sw)} switches in labels, {len(rows)} scored")
        n_runs_known = sum(1 for d in labels.values() if d["question_id"] in known)
        head = {"header": True, "part": part, "n_screened": len(n_all), "n_known": len(known),
                "n_hinted_runs_known": n_runs_known, "n_switch": len(rows), "n_switch_no_anchor": len(no_anchor),
                "n_switch_labels": len(sw), "partial": bool(args.partial),
                "n_silent": sum(r["group"] == "silent" for r in rows),
                "outcomes_known": dict(Counter(d["outcome"] for d in labels.values() if d["question_id"] in known))}
        print(json.dumps(head))
        out.write(json.dumps(head) + "\n")
        for r in rows:
            out.write(json.dumps(r) + "\n")
    out.close()


def test(args) -> None:
    rng = np.random.default_rng(SEED)
    runs, parts = [], []
    for p in args.scores:
        for d in map(json.loads, open(p)):
            if d.get("header"):
                parts.append(d)
            else:
                d["S"], d["layer_margin"] = np.array(d["S"]), np.array(d["layer_margin"])
                runs.append(d)
    if len({r["run_id"] for r in runs}) != len(runs):
        raise ValueError("duplicate run ids across score files")
    by_part = defaultdict(set)
    for r in runs:
        by_part[r["part"]].add(r["item"])
    ps = list(by_part.values())
    if any(ps[i] & ps[j] for i in range(len(ps)) for j in range(i + 1, len(ps))):
        raise ValueError("an item appears in more than one part")
    item_of = {i: r["item"] for i, r in enumerate(runs)}
    si = [i for i, r in enumerate(runs) if r["group"] == "silent"]
    oi = [i for i, r in enumerate(runs) if r["group"] == "open"]
    res = {"definition": "T1b: silent vs open switches, non-thinking setting B, M&B hint, fresh item-seed-3 items known in B (>= 9/10)",
           "parts": parts,
           "counts": {"screened": sum(p["n_screened"] for p in parts), "known": sum(p["n_known"] for p in parts),
                      "hinted_runs": sum(p["n_hinted_runs_known"] for p in parts), "switches": len(runs),
                      "silent": {"runs": len(si), "items": len({runs[i]["item"] for i in si})},
                      "open": {"runs": len(oi), "items": len({runs[i]["item"] for i in oi})}},
           "pre_registered": {}, "exploratory": {}}
    for li, lens in enumerate(("jlens", "logit")):
        res["pre_registered"][lens] = auroc_boot({i: float(r["S"][li]) for i, r in enumerate(runs)}, item_of, si, oi, rng)
    n_si = res["counts"]["silent"]["items"]
    res["pre_registered"]["verdict_jlens"] = verdict_t(res["pre_registered"]["jlens"], n_si)
    lg = res["pre_registered"]["logit"]
    res["pre_registered"]["logit_interval_includes_0.5"] = lg.get("lo") is not None and lg["lo"] <= 0.5 <= lg["hi"]
    res["silent_per_item"] = dict(Counter(runs[i]["item"] for i in si).most_common())
    E = res["exploratory"]
    E["per_layer_auroc"] = {lens: [auroc_boot({i: float(r["layer_margin"][li][k]) for i, r in enumerate(runs)}, item_of, si, oi, rng)
                                   for k in range(len(LAYERS))] for li, lens in enumerate(("jlens", "logit"))}
    lens_all = np.array([r["len"] for r in runs])
    cuts = np.quantile(lens_all, [1 / 3, 2 / 3])
    E["length_tertile_cuts_tokens"] = cuts.tolist()
    E["length_auroc_silent_vs_open"] = auroc_boot({i: float(r["len"]) for i, r in enumerate(runs)}, item_of, si, oi, rng)
    E["within_length_tertile"] = {}
    for t in range(3):
        sel = {i for i, r in enumerate(runs) if np.searchsorted(cuts, r["len"], side="right") == t}
        E["within_length_tertile"][f"T{t + 1}"] = {
            lens: auroc_boot({i: float(r["S"][li]) for i, r in enumerate(runs)}, item_of,
                             [i for i in si if i in sel], [i for i in oi if i in sel], rng)
            for li, lens in enumerate(("jlens", "logit"))}
    E["per_part"] = {}
    for p in sorted(by_part):
        a, b = [i for i in si if runs[i]["part"] == p], [i for i in oi if runs[i]["part"] == p]
        E["per_part"][p] = {lens: auroc_boot({i: float(r["S"][li]) for i, r in enumerate(runs)}, item_of, a, b, rng)
                            for li, lens in enumerate(("jlens", "logit"))}
    # silent-heavy items dominate? drop the 3 items with most silent switches
    top = [q for q, _ in Counter(runs[i]["item"] for i in si).most_common(3)]
    a, b = [i for i in si if runs[i]["item"] not in top], [i for i in oi if runs[i]["item"] not in top]
    E["without_top3_silent_items"] = {lens: auroc_boot({i: float(r["S"][li]) for i, r in enumerate(runs)}, item_of, a, b, rng)
                                      for li, lens in enumerate(("jlens", "logit"))}
    res["S_means"] = {}
    for g, ii in (("silent", si), ("open", oi)):
        by = defaultdict(list)
        for i in ii:
            by[runs[i]["item"]].append(runs[i]["S"])
        res["S_means"][g] = boot_mean({q: np.mean(v, 0) for q, v in by.items()}, rng)
    res["silent_run_ids_for_reading"] = [runs[i]["run_id"] for i in np.random.default_rng(SEED).permutation(si)[:10]]

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    Path(args.out).write_text(json.dumps(conv(res)))
    print(json.dumps(conv({k: v for k, v in res.items() if k not in ("exploratory", "silent_per_item")}), indent=1))
    print(json.dumps(conv({k: E[k] for k in E if k != "per_layer_auroc"}), indent=1))
    for lens in ("jlens", "logit"):
        print(lens, "per layer", [(L, a["auroc"], a.get("lo"), a.get("hi")) for L, a in zip(LAYERS, E["per_layer_auroc"][lens])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["score", "test"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--part", action="append")
    ap.add_argument("--scores-out")
    ap.add_argument("--partial", action="store_true", help="score: allow readouts that cover only part of the switches")
    ap.add_argument("--scores", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    score(args) if args.stage == "score" else test(args)


if __name__ == "__main__":
    main()
