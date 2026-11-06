"""F1-T: which F1 cells were inconclusive and how many more items they need (CPU, stdlib).
Item-bootstrap half-width h of the 90% interval scales as 1/sqrt(items). Equivalence within
+-0.10 needs h <= 0.10 - |true - 0.5|. Two assumptions: true = 0.5, and true = the observed point.
    python scripts/lens/f1t_power.py
"""
import json, math
R = "docs/experiments/firming-f1-f2/data/"
allc = json.load(open(R + "f1_all.json"))["cells"]
cases = json.load(open(R + "f1_cases.json"))
ITEMS = {"wording": 50, "c3": 40, "t2_followers_still_know_vs_persuaded": 54, "rcl_claimed_answer_key": None,
         "grader_q14": 69, "hack_problem_means": 59, "ea27_smoke": 19}
out = {}
for name in ("wording", "c3", "t2_followers_still_know_vs_persuaded", "rcl_claimed_answer_key", "grader_q14",
             "hack_problem_means", "ea27_smoke"):
    first = next(iter(allc[name].values()))
    n_items = cases[name]["n_items"] if name in cases else first.get("n_clusters") or ITEMS[name]
    out[name] = {"items_old": n_items, "positions": {}}
    for pos, cell in allc[name].items():
        j = cell["jlens"]; lo, hi = j["ci90"]; h = (hi - lo) / 2; d = abs(j["point"] - .5)
        def need(true_d):
            room = .10 - true_d
            return None if room <= 0.01 else math.ceil(n_items * (h / room) ** 2)
        n80 = math.ceil(n_items * (h / 0.056) ** 2)  # 80% chance to pass TOST if the truth is exactly 0.5 (needs h <= 0.056)
        out[name]["positions"][pos] = {"items_total_for_80pct_power_if_true_is_chance": n80, "point": round(j["point"], 3), "ci90": [round(x, 3) for x in j["ci90"]],
            "equivalent_now": j["equiv_chance"], "h90": round(h, 3),
            "items_total_needed_if_true_is_chance": need(0.0), "items_total_needed_if_true_is_observed": need(d),
            }
json.dump(out, open("docs/experiments/firming-f1-f2/data/f1t_power.json", "w"), indent=1)
for k, v in out.items():
    print(k, v["items_old"])
    for p, c in v["positions"].items():
        print("  ", p, c["point"], c["ci90"], "h", c["h90"], "equiv" if c["equivalent_now"] else "-", "n80", c["items_total_for_80pct_power_if_true_is_chance"], "need(null)", c["items_total_needed_if_true_is_chance"], "need(obs)", c["items_total_needed_if_true_is_observed"])
