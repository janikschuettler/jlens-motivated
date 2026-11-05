"""Print the big markdown table of docs/experiments/jlens-vs-competitors/report.md from
docs/experiments/jlens-vs-competitors/data/jlens_vs_competitors.json (stdlib only). * = copied from an earlier report."""
import json, sys
d = json.load(open("docs/experiments/jlens-vs-competitors/data/jlens_vs_competitors.json"))
POS = [("options", "option words"), ("last_prompt", "last prompt token"), ("mid", "mid reasoning"),
       ("think_end", "end of reasoning"), ("pre_answer", "answer position"), ("whole", "whole run")]
COLS = ["jlens", "logit", "probe", "blackbox", "monitor", "length"]
def f(v):
    if v is None: return ""
    a = v.get("auroc")
    if a is None: return ""
    s = f"{a:.2f}"
    if v.get("lo") is not None and v.get("hi") is not None:
        s += f" [{v['lo']:.2f}, {v['hi']:.2f}]"
    if v.get("source", "new") != "new": s += " *"
    return s
num = 0
print("| Case, position | J-lens | logit lens | linear probe | black box (model's own answer) | CoT keyword monitor | length |")
print("|---|---|---|---|---|---|---|")
for cid, meta in d["cases"].items():
    t = d["table"][cid]
    print(f"| **{meta['label']}** ({meta['model']}) | | | | | | |")
    for p, pl in POS:
        cells = []
        for c in COLS:
            v = t.get(f"{p}|{c}")
            if c == "monitor" and f"{p}|monitor_intent" in t:
                a = f(v); b = f(t[f"{p}|monitor_intent"])
                cells.append(f"{a} (mentions bug); {b} (intent words)")
                continue
            if v is None and c in ("jlens", "logit") and f"{p}|{c}_concept" in t:
                cells.append(f(t[f"{p}|{c}_concept"]) + " (grader words)")
                continue
            if v is not None and p == "pre_answer" and meta["answer_differs"] and v.get("auroc") is not None and min(v["auroc"], 1 - v["auroc"]) < 0.12:
                s = f"{v['auroc']:.2f} (reads the answer)"
                if f"{p}|{c}_concept" in t:
                    s += f"; grader words {f(t[f'{p}|{c}_concept'])}"
                cells.append(s)
                continue
            s = f(v)
            if v is not None and "earlier" in v:
                e = v["earlier"]
                if cid != "silent" and e.get("auroc") is not None and abs(e["auroc"] - v["auroc"]) < 0.015:
                    s += " (= earlier report)"
                else:
                    s += f"; earlier {f(e)}"
            if v is not None and "earlier_old_items" in v:
                s = (s + "; " if s and s != "" else "") + f"old items {f(v['earlier_old_items'])}"
            if f"{p}|{c}_concept" in t:
                s += f"; grader words {f(t[f'{p}|{c}_concept'])}"
            cells.append(s)
        if any(cells):
            print(f"| {pl} | " + " | ".join(cells) + " |")
