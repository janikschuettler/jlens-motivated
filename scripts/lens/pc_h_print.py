import json, sys
r = json.load(open(sys.argv[1]))
f = lambda x: f"{x['auroc']:.2f} [{x['ci95'][0]:.2f},{x['ci95'][1]:.2f}]"
g = lambda x: f"{x['est']:+.2f} [{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}]"
for k in ("keyword", "strict_yes_vs_no", "yesmaybe_vs_no"):
    v = r[k]; print(k, v["n_prompts"], v["n_aware"], v["n_unaware"], v["mean_window_len"])
    for sec in ("primary", "last32", "first64", "matched_first", "matched_last", "residualised"):
        print(" ", sec, {n: f(x) for n, x in v[sec].items() if isinstance(x, dict) and "auroc" in x})
    print("  diffs", {n: g(x) for n, x in v["primary"].items() if "est" in x})
    print("  band", {a: [round(b, 2) for b in c.values()] for a, c in v["random_band"].items()})
