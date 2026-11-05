"""hint-label-validation, step 1 (CPU, envs/judge): build the population table and the stratified sample.

Population: Qwen3-8B thinking-mode runs on known items. Sources (copies read-only from arena9-octavia into
$HLV_DATA): factual = gap_factorial/factorial + w2/wording (hinted thinking arms, unhinted thinking arms);
moral = moral_hunt nudge+nudge2 (mayor backup), w2/moral_nudge mb_expert arm.
Keyword rule ("mentions the hint") = any of w2_silent_scan.BROAD or the run's own mention patterns matches the CoT.
Writes $HLV_DATA/population.jsonl (no text) and $HLV_DATA/sample.jsonl (with prompt, CoT, response).
Pre-registered: docs/experiments/hint-label-validation/preregistration.md
"""
import importlib.util, json, os, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

from jmr.label import split_completion

D = Path(os.environ["HLV_DATA"])
spec = importlib.util.spec_from_file_location("ss", "scripts/explore/w2_silent_scan.py")
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
BROAD = [re.compile(x, re.I) for x in ss.BROAD]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def known(path, key="acc"):
    out = {}
    for r in jl(path):
        out[r["id"]] = (r.get("acc") or {}).get("A")
    return out


fact_known = {k for k, v in known(D / "known_items.jsonl").items() if (v or 0) >= 0.9}
moral_ids = {r["id"] for r in jl(D / "mh/known.jsonl")}
pop = []  # one dict per thinking run


def load(dirpath, domain, src):
    lab = {r["run_id"]: r for r in jl(dirpath / "labels.jsonl")}
    for r in jl(dirpath / "runs.jsonl"):
        L = lab.get(r["run_id"])
        if not L or not L.get("thinking"):
            continue
        yield r, L


for src, path, domain in [("gf", D / "gf/factorial", "factual"), ("wording", D / "w2/wording", "factual"),
                          ("mh_nudge", D / "mh/nudge", "moral"), ("mh_nudge2", D / "mh/nudge2", "moral"),
                          ("w2_moral", D / "w2/moral_nudge", "moral")]:
    for r, L in load(path, domain, src):
        arm = L["arm"]
        if src == "w2_moral" and arm != "mb_expert":
            continue
        if domain == "factual" and L["question_id"] not in fact_known:
            continue
        hinted = "unhinted" not in arm and arm not in ("null_control", "V0_bug")
        o = L["outcome"]
        if o in ("excluded", "other_wrong") or L.get("finish_reason", "stop") != "stop":
            continue
        kind = ("switch" if o in ("switch", "followed") else "resisted" if o in ("resisted", "correct") and hinted
                else "nohint" if not hinted and o in ("correct", "wrong") else None)
        if kind is None:
            continue
        cot, resp = split_completion(r["completion_text"])
        if resp is None:
            continue
        own = [re.compile(x, re.I) for x in (L.get("mentions") or [])]
        hits = [bool(x.search(cot)) for x in BROAD + own]
        nh = sum(hits)
        pop.append(dict(run_id=r["run_id"], domain=domain, src=src, arm=arm, item=L["question_id"], kind=kind,
                        kw_mention=nh > 0, n_hits=nh, cot_chars=len(cot)))
        r["_cot"], r["_resp"] = cot, resp
        if kind != "nohint" or True:
            pop[-1]["_row"] = r

# strata
def stratum(p):
    if p["kind"] == "switch":
        return "switch_silent" if p["n_hits"] == 0 else "switch_weak" if p["n_hits"] <= 5 else "switch_strong"
    return p["kind"] + ""

for p in pop:
    p["stratum"] = stratum(p) + "|" + p["domain"]
cnt = Counter(p["stratum"] for p in pop)
print(json.dumps(dict(sorted(cnt.items())), indent=1))
(D / "population.jsonl").write_text("".join(json.dumps({k: v for k, v in p.items() if k != "_row"}) + "\n" for p in pop))

# sample (seed 0)
rng = random.Random(0)
quota = {"switch_silent|factual": 10**9, "switch_silent|moral": 10**9,
         "switch_weak|factual": 40, "switch_weak|moral": 40, "switch_strong|factual": 50, "switch_strong|moral": 50,
         "resisted|factual": 50, "resisted|moral": 50, "nohint|factual": 100}
sample = []
for st, q in quota.items():
    cand = [p for p in pop if p["stratum"] == st]
    rng.shuffle(cand)
    for p in cand[:q]:
        r = p["_row"]
        user = [m["content"] for m in r["messages"] if m["role"] == "user"][-1]
        sysm = [m["content"] for m in r["messages"] if m["role"] == "system"]
        L_hint = r.get("hint_letter") or r.get("letter")
        corr = r["correct_letter"]
        if L_hint is None or L_hint == corr:  # unhinted: random wrong letter as the control "candidate hint"
            letters = [x for x in (r.get("letters") or ["A", "B", "C", "D"]) if x != corr]
            L_hint = rng.choice(letters)
        sample.append(dict(run_id=p["run_id"], stratum=st, domain=p["domain"], kind=p["kind"], item=p["item"], arm=p["arm"],
                           kw_mention=p["kw_mention"], n_hits=p["n_hits"], system=sysm[0] if sysm else "", prompt=user,
                           cand_letter=L_hint, correct_letter=corr, cot=r["_cot"].strip(), response=r["_resp"].strip()))
rng.shuffle(sample)
(D / "sample.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sample))
print("sample", len(sample), Counter(s["stratum"] for s in sample))
