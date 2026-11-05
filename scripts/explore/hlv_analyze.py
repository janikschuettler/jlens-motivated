"""hint-label-validation, step 3 (numpy; any env): agreement of two judges with the keyword rule and the implied silent rate.
  HLV_DATA=... python scripts/explore/hlv_analyze.py --out docs/experiments/hint-label-validation/data/hlv.json
"""
import argparse, json, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

D = Path(os.environ["HLV_DATA"])
J = {"flash_lite": "hlv_judge_google_gemini-2.5-flash-lite.jsonl", "gpt5_nano": "hlv_judge_openai_gpt-5-nano.jsonl"}


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if len(a) == 0:
        return None
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def conf(kw, jd):
    kw, jd = np.asarray(kw, bool), np.asarray(jd, bool)
    return {"kw_yes_judge_yes": int((kw & jd).sum()), "kw_yes_judge_no": int((kw & ~jd).sum()),
            "kw_no_judge_yes": int((~kw & jd).sum()), "kw_no_judge_no": int((~kw & ~jd).sum())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    sample = {s["run_id"]: s for s in jl(D / "sample.jsonl")}
    pop = jl(D / "population.jsonl")
    N = Counter(p["stratum"] for p in pop)
    jud = {}
    for k, f in J.items():
        jud[k] = {r["run_id"]: r["judge"] for r in jl(D / f)}
    ids = [i for i in sample if all(i in jud[k] for k in jud)]
    print(len(sample), "sampled;", len(ids), "judged by both")
    men = {k: {i: jud[k][i]["mentions_hint"] == "yes" for i in ids} for k in jud}
    rel = {k: {i: jud[k][i]["relies_on_hint"] == "yes" for i in ids} for k in jud}
    kw = {i: sample[i]["kw_mention"] for i in ids}
    strata = sorted({sample[i]["stratum"] for i in ids})
    out = {"n_sample": len(sample), "n_judged_both": len(ids), "population": dict(N), "by_stratum": {}, "agreement": {}, "implied_silent": {}}
    for st in strata:
        ii = [i for i in ids if sample[i]["stratum"] == st]
        out["by_stratum"][st] = {"n": len(ii), "N_pop": N[st], "keyword_mention": int(sum(kw[i] for i in ii)),
                                 **{f"{k}_mention": int(sum(men[k][i] for i in ii)) for k in jud},
                                 **{f"{k}_relies": int(sum(rel[k][i] for i in ii)) for k in jud},
                                 "both_say_no_mention": int(sum((not men["flash_lite"][i]) and (not men["gpt5_nano"][i]) for i in ii))}
    var_set = [i for i in ids if sample[i]["kind"] != "switch" or sample[i]["stratum"].startswith("switch_silent")]  # strata where keyword label varies
    for k in jud:
        out["agreement"][f"keyword_vs_{k}"] = {
            "all": {"kappa": kappa([kw[i] for i in ids], [men[k][i] for i in ids]), **conf([kw[i] for i in ids], [men[k][i] for i in ids])},
            "resisted_unhinted_silent": {"n": len(var_set), "kappa": kappa([kw[i] for i in var_set], [men[k][i] for i in var_set]),
                                         **conf([kw[i] for i in var_set], [men[k][i] for i in var_set])},
            "switches_only": conf([kw[i] for i in ids if sample[i]["kind"] == "switch"], [men[k][i] for i in ids if sample[i]["kind"] == "switch"])}
    f, g = "flash_lite", "gpt5_nano"
    out["agreement"]["judge_vs_judge_mention"] = {"kappa": kappa([men[f][i] for i in ids], [men[g][i] for i in ids]),
                                                  "percent_agree": float(np.mean([men[f][i] == men[g][i] for i in ids])), **conf([men[f][i] for i in ids], [men[g][i] for i in ids])}
    out["agreement"]["judge_vs_judge_relies"] = {"kappa": kappa([rel[f][i] for i in ids], [rel[g][i] for i in ids]),
                                                 "percent_agree": float(np.mean([rel[f][i] == rel[g][i] for i in ids]))}
    nh = [i for i in ids if sample[i]["kind"] == "nohint"]
    out["unhinted_false_positive_mention"] = {k: [int(sum(men[k][i] for i in nh)), len(nh)] for k in jud}
    # implied silent rate among switches, per domain
    rng = np.random.default_rng(0)
    for dom in ("factual", "moral"):
        sts = [s for s in strata if s.startswith("switch") and s.endswith(dom)]
        # per stratum: list of (item, flag) for each variant
        def flags(i):
            return {"either_no": (not men[f][i]) or (not men[g][i]), "both_no": (not men[f][i]) and (not men[g][i]),
                    "flash_lite_no": not men[f][i], "gpt5_nano_no": not men[g][i]}
        res = {}
        for var in ("either_no", "both_no", "flash_lite_no", "gpt5_nano_no"):
            def est(draw=None):
                num = den = 0.0
                for st in sts:
                    ii = [i for i in ids if sample[i]["stratum"] == st]
                    if draw is not None:
                        items = sorted({sample[i]["item"] for i in ii}); pick = rng.choice(items, len(items))
                        byit = defaultdict(list)
                        for i in ii: byit[sample[i]["item"]].append(i)
                        ii = [i for it in pick for i in byit[it]]
                    if not ii: continue
                    p = np.mean([flags(i)[var] for i in ii])
                    num += N[st] * p; den += N[st]
                return num / den if den else float("nan")
            bs = [est(True) for _ in range(1000)]
            res[var] = {"rate": est(), "ci95": [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))]}
        out["implied_silent"][dom] = {"population_switches": sum(N[s] for s in sts), "rates": res}
    # unweighted sampled-switch counts with exact (Clopper-Pearson) 95% upper bounds
    from math import comb
    def cp_upper(k, n):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            tail = sum(comb(n, j) * mid**j * (1 - mid) ** (n - j) for j in range(k + 1))
            lo, hi = (mid, hi) if tail > 0.025 else (lo, mid)
        return hi
    out["sampled_switch_no_mention"] = {}
    for dom in ("factual", "moral"):
        ii = [i for i in ids if sample[i]["kind"] == "switch" and sample[i]["domain"] == dom]
        out["sampled_switch_no_mention"][dom] = {"n_sampled_switches": len(ii), **{
            k: {"k": int(sum(not men[k][i] for i in ii)), "upper95": cp_upper(int(sum(not men[k][i] for i in ii)), len(ii))} for k in jud},
            "both": {"k": int(sum((not men["flash_lite"][i]) and (not men["gpt5_nano"][i]) for i in ii)),
                     "upper95": cp_upper(int(sum((not men["flash_lite"][i]) and (not men["gpt5_nano"][i]) for i in ii)), len(ii))}}
    # reliance among switches
    sw = [i for i in ids if sample[i]["kind"] == "switch"]
    out["relies_on_hint_among_sampled_switches"] = {k: [int(sum(rel[k][i] for i in sw)), len(sw)] for k in jud}
    out["relies_among_resisted"] = {k: [int(sum(rel[k][i] for i in ids if sample[i]["kind"] == "resisted")), sum(sample[i]["kind"] == "resisted" for i in ids)] for k in jud}
    Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
