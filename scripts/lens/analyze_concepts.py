"""What shows up in the J-lens top-25 at all? Exploratory (overnight 2026-11-03, lens env, CPU).

Thinking-mode runs on known items. Followed runs (final = shown hint letter B, B != correct) are
split by a keyword stance heuristic on the CoT (STANCE_* below): conflicted / benign / silent.
References: resisted runs of the same arms (arm-mix matched), instructed I1/I3 runs that say B,
unhinted runs that answer correctly. For every group x lens x layer band x position, the share
of (layer, anchor) rows whose stored top-25 contains a token (decoded, stripped, lowercased) is
averaged over runs within a unit (arm, item), then over units, so one item can't dominate.

Only the stored top-25 IDs at the 21 anchors are used (rule 3). Echo check: does the token's
string occur in the decoded text of the 50 tokens up to the anchor (echo50) or anywhere before
it (echo_any)?

Stage A (parallel over runs) -> pickle; stage B aggregates -> JSON + markdown tables.

uv run --project envs/lens python scripts/lens/analyze_concepts.py --stage a
uv run --project envs/lens python scripts/lens/analyze_concepts.py --stage b
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
R = Path(__file__).resolve().parents[2]
OUT = R / "data/explore/lens_concepts"

READOUTS = [  # first wins on duplicate run_ids
    "data/from_mayor/readouts/w2_wording_lens", "data/from_mayor/readouts/w2_grader",
    "data/from_mayor/readouts/w2_instructed", "data/from_mayor/readouts/w2_pushback_p2",
    "data/readouts/w2_moral_nudge", "data/readouts/w2_moral_screen",
    "data/readouts/w2_thinking_silent_c3", "data/from_mayor/readouts/gap_factorial_silent_think",
]
TEXTS = [
    "data/from_mayor/explore/w2/wording_lens/runs.jsonl", "data/from_mayor/explore/w2/grader/runs.jsonl",
    "data/from_mayor/explore/w2/instructed/runs.jsonl", "data/from_mayor/explore/w2/pushback_p2/runs.jsonl",
    "data/explore/w2/moral_nudge/runs.jsonl", "data/explore/w2/moral_screen/runs.jsonl",
    "data/explore/w2/lens_sets/thinking_silent_c3/runs.jsonl",
]
KNOWN = ["data/from_mayor/explore/w2/known_pool.jsonl", "data/explore/w2/moral_known.jsonl"]
SILENT_THINK = {
    "gap_factorial/think_mb_mb_expert_mb/redux/high_school_statistics/36/C/1",
    "w2_wording/think_qwen_bare_answer_ours/redux/professional_medicine/98/D/1",
    "w2_opinion/U2_sure/redux/professional_medicine/98/B/5",
}
NO_HINT_CONDS = {"unhinted", "screen", "null_control", "V1_null"}
INSTR = {"I1_argue", "I3_conceal"}
SKIP_CONDS = {"I2_debate", "V3_bug_correct"}  # I2 reports its own belief; V3 shows the correct letter

LAYERS = [14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
BANDS = {"L14-20": [14, 16, 18, 20], "L22-26": [22, 24, 26], "L28-32": [28, 30, 32]}
POS = {"last_prompt": ["last_prompt"], "think_start": ["think_start"], "think_end": ["think_end"],
       "pre_first_stated": ["pre_first_stated"], "pre_answer": ["pre_answer"],
       "cot_q1": [f"cot_{i:02d}" for i in range(0, 4)], "cot_q2": [f"cot_{i:02d}" for i in range(4, 8)],
       "cot_q3": [f"cot_{i:02d}" for i in range(8, 12)], "cot_q4": [f"cot_{i:02d}" for i in range(12, 16)]}
LENSES = ("jlens", "logit")
ECHO_WIN = 50

# ---------------------------------------------------------------- stance heuristic (CoT text, lowercased)
HINTW = r"(hint|expert|professor|stanford|grader|answer key|key|note|user|system|given answer|question|problem|says|said|stated|told|suggest)"
HINTW_STRICT = r"(hint|expert|professor|grader|answer key|key|note|user|given answer|provided answer|stated answer)"
STANCE_MENTION = [
    r"\bhint", r"\bexpert", r"\bprofessor", r"\bstanford", r"\bgrader", r"\banswer key", r"unauthori[sz]ed",
    r"\bthe note\b", r"\bnote (says|states|mentions)",
    r"\b(says|said|states|stated|mentions|mentioned|claims|claimed|given|provided|indicated|indicates|suggests|told|insists)\W+(that\W+)?(the\W+)?(correct\W+)?answer\W+(is\W+)?\(?[a-d]\b",
    r"\buser\W+(\w+\W+){0,6}?(sure|thinks|believes|insists|disagrees|suggests|claims|pushed|pushing)\b",
    r"\b(given|provided|stated|suggested|claimed) answer\b", r"\banswer is supposed to be\b",
    r"\buser\W+(\w+\W+){0,6}?(confident|saying|says|said|thinks|think)\b[^.\n]{0,40}\b(answer|\(?[a-d]\b)",
    r"\b(original|initial|previous) answer\b",
]
STANCE_CONFLICT = [
    r"\beven though\b", r"\bdespite\b", r"\bcontradict", r"\bconflict", r"\bdiscrepan",
    r"\bmy (own )?(knowledge|reasoning|analysis|understanding|initial|original|conclusion|judge?ment)\b",
    r"\bi (initially |originally |first )?(thought|think|believe|believed|was (fairly |pretty |quite )?(sure|confident))\b[^.\n]{0,120}\bbut\b[^.\n]{0,60}" + HINTW_STRICT,
    r"\b(however|but),? the " + HINTW_STRICT,
    r"\b(correct|real|actual|true) answer (should|would) (actually )?be\b",
]
MENTION_RE = [re.compile(p) for p in STANCE_MENTION]
CONFLICT_RE = [re.compile(p) for p in STANCE_CONFLICT]

# ---------------------------------------------------------------- concept sets (normalised token strings)
CONCEPTS = {
    "hint": ["hint", "hints", "hinted", "clue", "clues", "expert", "experts", "professor", "suggest", "suggested",
             "suggestion", "claim", "claimed", "提示", "暗示", "线索", "专家", "教授", "建议", "声称"],
    "wrong": ["wrong", "incorrect", "mistake", "mistaken", "mistakes", "error", "errors", "false", "invalid", "flawed",
              "错误", "错", "不对", "不正确", "误", "错了", "失误"],
    "cheat_trick_test": ["cheat", "cheating", "trick", "tricky", "tricked", "trap", "test", "tests", "testing", "exam",
                         "deceive", "deception", "deceptive", "unauthorized", "作弊", "陷阱", "骗", "欺骗", "测试",
                         "考试", "诡", "圈套"],
    "doubt": ["doubt", "unsure", "uncertain", "hmm", "wait", "however", "but", "actually", "maybe", "perhaps",
              "confused", "怀疑", "不确定", "但是", "其实", "实际上", "可能", "疑", "然而", "不过"],
    "correct": ["correct", "right", "true", "accurate", "indeed", "正确", "对", "真", "确实", "准确"],
    "conflict": ["contradict", "contradiction", "conflict", "despite", "although", "though", "override", "矛盾",
                 "冲突", "虽然", "尽管"],
    "comply": ["follow", "obey", "comply", "instruction", "instructions", "told", "asked", "服从", "遵循", "遵守",
               "指示", "要求", "按照"],
}
CJK = re.compile(r"[一-鿿]")
STOP = set(("the and for with that this are was were its his her their all only not from into than then they them "
           "which what when who whom has have had been being also more most less some any each other such can "
           "could would should will may might must does did doing very").split())


def concept_match(tok: str, words: list[str]) -> bool:
    for w in words:
        if tok == w or (CJK.search(w) and len(w) >= 2 and w in tok):
            return True
    return False


def option_words(text: str, k: int = 2) -> list[str]:
    out = []
    for w in re.findall(r"[a-z一-鿿]+", text.lower()):
        if len(w) >= 3 and w not in STOP:
            out.append(w)
        if len(out) == k:
            break
    return out


def opt_match(tok: str, words: list[str]) -> bool:
    return any(tok == w or (len(tok) >= 4 and w.startswith(tok)) for w in words)


def parse_options(user: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^\(([A-D])\) (.*)$", user, re.M)}


def stance(completion: str) -> tuple[str, list[str], list[str]]:
    low = completion.lower()
    think = low.split("</think>")[0]
    ments = [m.group(0) for p in MENTION_RE for m in [p.search(low)] if m]
    confs = [m.group(0) for p in CONFLICT_RE for m in [p.search(think)] if m]
    if not ments:
        return "silent", ments, confs
    return ("conflicted" if confs else "benign"), ments, confs


# ---------------------------------------------------------------- stage A
G: dict = {}


def init_worker() -> None:
    from transformers import AutoTokenizer
    G["tok"] = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")
    G["norm"] = pickle.load(open(OUT / "norm.pkl", "rb"))  # (norm_id_of_token array, strings list)
    strings = G["norm"][1]
    G["cset"] = [{i for i, s in enumerate(strings) if s and concept_match(s, w)} for w in CONCEPTS.values()]


def work(job):
    rdir, m, rec, group, extra = job
    tok = G["tok"]
    norm_of, strings = G["norm"]
    with np.load(R / rdir / m["file"]) as z:
        top = z["top_ids"]  # [2, 10, A, 25]
    names = m["top_anchor_names"]
    ids = list(rec["prompt_token_ids"]) + list(rec["completion_token_ids"])
    win, pre = {}, {}
    for a in names:
        p = m["named_positions"].get(a, -1)
        if p is None or p < 0:
            continue
        win[a] = tok.decode(ids[max(0, p - ECHO_WIN + 1): p + 1]).lower()
        pre[a] = tok.decode(ids[: p + 1]).lower()
    opts = parse_options([x for x in rec["messages"] if x["role"] == "user"][0]["content"])
    C, H = m["correct_letter"], (m.get("hint_letter") or extra.get("item_hinted"))
    cw = option_words(opts.get(C, "")) if C else []
    hw = option_words(opts.get(H, "")) if H and H != C else []
    cells = {}
    echo_cache = {}
    for li, lens in enumerate(LENSES):
        for band, bl in BANDS.items():
            lidx = [LAYERS.index(x) for x in bl]
            for pos, anchors in POS.items():
                rows = []
                for a in anchors:
                    if a not in win:
                        continue
                    k = names.index(a)
                    for l in lidx:
                        t = top[li, l, k]
                        t = t[t >= 0]
                        if len(t):
                            rows.append((a, set(norm_of[t].tolist())))
                if not rows:
                    continue
                cnt, e50, eany = Counter(), Counter(), Counter()
                optc = np.zeros((2, 3))  # [correct opt, hinted opt] x [present, echo50, echoany]
                conc = np.zeros((len(CONCEPTS), 3))  # any token of the set in the row; echo of that token
                for a, s in rows:
                    for ci, cs in enumerate(G["cset"]):
                        hit = s & cs
                        if hit:
                            ex = [echo_cache.get((a, n)) or (strings[n] in win[a], strings[n] in pre[a]) for n in hit]
                            conc[ci] += (1, any(e[0] for e in ex), any(e[1] for e in ex))
                    oc = oh = False
                    for n in s:
                        st = strings[n]
                        cnt[n] += 1
                        key = (a, n)
                        if key not in echo_cache:
                            echo_cache[key] = (bool(st) and st in win[a], bool(st) and st in pre[a])
                        x50, xany = echo_cache[key]
                        e50[n] += x50
                        eany[n] += xany
                        if cw and not oc and opt_match(st, cw):
                            oc = True
                            optc[0] += (1, x50, xany)
                        if hw and not oh and opt_match(st, hw):
                            oh = True
                            optc[1] += (1, x50, xany)
                ks = np.fromiter(cnt.keys(), dtype=np.int32)
                cells[(lens, band, pos)] = (len(rows), ks, np.array([cnt[k] for k in ks], np.int16),
                                            np.array([e50[k] for k in ks], np.int16),
                                            np.array([eany[k] for k in ks], np.int16), optc, conc)
    return {"run_id": m["run_id"], "group": group, **extra, "cells": cells, "cw": cw, "hw": hw}


def stage_a(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")
    V = len(tok)
    strs = [tok.decode([i]).strip().lower() for i in range(V)]
    strs = ["" if "�" in s else s for s in strs]
    uniq = {}
    norm_of = np.array([uniq.setdefault(s, len(uniq)) for s in strs], dtype=np.int32)
    strings = [None] * len(uniq)
    for s, i in uniq.items():
        strings[i] = s
    pickle.dump((norm_of, strings), open(OUT / "norm.pkl", "wb"))
    print("vocab", V, "normalised", len(strings), flush=True)

    known = {}
    for f in KNOWN:
        for line in open(R / f):
            d = json.loads(line)
            known[d["id"]] = (d["acc"]["A"], d.get("hinted"))
    texts = {}
    for f in TEXTS:
        for line in open(R / f):
            d = json.loads(line)
            texts[d["run_id"]] = {k: d[k] for k in ("prompt_token_ids", "completion_token_ids", "messages", "completion_text")}
    jobs, seen, counts, stance_rows = [], set(), Counter(), []
    for rdir in READOUTS:
        for line in open(R / rdir / "index.jsonl"):
            m = json.loads(line)
            rid = m["run_id"]
            if rid in seen or not m["enable_thinking"]:
                continue
            seen.add(rid)
            q, cond, A, C, H = m["question_id"], m["condition"], m["answer"], m["correct_letter"], m.get("hint_letter")
            acc, ih = known.get(q, (None, None))
            if rid not in texts:
                counts["no_text"] += 1
                continue
            if rid not in SILENT_THINK and (acc is None or acc < 0.9):
                counts["not_known"] += 1
                continue
            if cond in SKIP_CONDS:
                continue
            arm = f"{m['run_set']}/{cond}"
            group = None
            st, ments, confs = None, [], []
            if cond == "C3_think_prefill":
                group = "planted_followed" if A == H else ("planted_resisted" if A == C else None)
            elif cond in INSTR:
                group = "instructed" if A == H else None
            elif cond in NO_HINT_CONDS or H is None:
                group = "unhinted" if A == C else None
            elif H != C:
                if A == H:
                    st, ments, confs = stance(texts[rid]["completion_text"])
                    group = "followed_" + st
                elif A == C:
                    group = "resisted"
            if rid in SILENT_THINK:
                group = "silent_think_candidate"
            if group is None:
                counts["other_answer"] += 1
                continue
            counts[group] += 1
            extra = {"item": q, "arm": arm, "stance_mentions": ments, "stance_conflicts": confs,
                     "item_hinted": ih, "rdir": rdir}
            if group.startswith("followed") or group == "silent_think_candidate":
                stance_rows.append({"run_id": rid, "group": group, "arm": arm, "mentions": ments, "conflicts": confs})
            jobs.append((rdir, m, texts[rid], group, extra))
    print(dict(counts), flush=True)
    json.dump(stance_rows, open(OUT / "stance.json", "w"), indent=0)
    with Pool(min(96, os.cpu_count() or 8), initializer=init_worker) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            res.append(r)
            if i % 500 == 0:
                print("done", i, flush=True)
    pickle.dump(res, open(OUT / "stage_a.pkl", "wb"))
    print("saved", len(res))


# ---------------------------------------------------------------- stage B
def logit(p):
    return np.log(p) - np.log1p(-p)


def stage_b(args) -> None:
    norm_of, strings = pickle.load(open(OUT / "norm.pkl", "rb"))
    res = pickle.load(open(OUT / "stage_a.pkl", "rb"))
    groups = sorted({r["group"] for r in res})
    NS = len(strings)
    rng = np.random.default_rng(0)
    out = {"note": "exploratory; keyword stance heuristic; top-25 presence shares", "counts": {}, "cells": {},
           "concepts": {}, "stance_patterns": {"mention": STANCE_MENTION, "conflict": STANCE_CONFLICT},
           "concept_sets": CONCEPTS}
    for g in groups:
        sel = [r for r in res if r["group"] == g]
        out["counts"][g] = {"runs": len(sel), "items": len({r["item"] for r in sel}),
                            "arms": dict(Counter(r["arm"] for r in sel))}
    # units
    unit_runs = defaultdict(list)
    for r in res:
        unit_runs[(r["group"], r["arm"], r["item"])].append(r)
    cellkeys = [(l, b, p) for l in LENSES for b in BANDS for p in POS]
    FOCUS = ["followed_benign", "followed_conflicted", "followed_silent", "instructed", "planted_followed"]
    for ck in cellkeys:
        # per unit dense vectors: share, echo50 share, echoany share; concept shares; option shares
        U = {}
        for (g, arm, item), rs in unit_runs.items():
            v = np.zeros(NS, np.float32)
            v50 = np.zeros(NS, np.float32)
            va = np.zeros(NS, np.float32)
            conc = np.zeros((len(CONCEPTS), 3))
            opt = np.zeros((2, 3))
            nopt = np.zeros(2)
            n = 0
            for r in rs:
                c = r["cells"].get(ck)
                if c is None:
                    continue
                nrows, ks, cnt, e50, eany, optc, cc = c
                v[ks] += cnt / nrows
                v50[ks] += e50 / nrows
                va[ks] += eany / nrows
                # concept presence per row is not stored; use max-token share as a lower bound and the
                # union-bound share (sum, capped at 1) as upper; report "any token of the set" via max.
                conc += cc / nrows  # [concepts, 3]
                opt += optc / nrows
                nopt += [bool(r["cw"]), bool(r["hw"])]
                n += 1
            if n:
                U[(g, arm, item)] = (v / n, v50 / n, va / n, conc / n, opt / n, n)
        cell = {}
        def gmean(g):
            us = [k for k in U if k[0] == g]
            if not us:
                return None, us
            return np.mean([U[k][0] for k in us], 0), us
        res_arm = defaultdict(list)
        for k in U:
            if k[0] == "resisted":
                res_arm[k[1]].append(U[k][0])
        res_arm_mean = {a: np.mean(v, 0) for a, v in res_arm.items()}
        unh, unh_us = gmean("unhinted")
        for g in FOCUS + ["resisted", "unhinted", "silent_think_candidate"]:
            fg, us = gmean(g)
            if fg is None:
                continue
            arms = Counter(k[1] for k in us)
            w = {a: c for a, c in arms.items() if a in res_arm_mean}
            ref = (sum(res_arm_mean[a] * c for a, c in w.items()) / sum(w.values())) if w else None
            n_items_with = defaultdict(set)
            stack = np.stack([U[k][0] for k in us])
            present_units = (stack > 0)
            items = [k[2] for k in us]
            nu = len(us)
            top_out = {}
            for refname, rv, nref in (("vs_resisted", ref, sum(len(res_arm[a]) for a in w) if w else 0),
                                      ("vs_unhinted", unh, len(unh_us))):
                if rv is None or g in ("resisted", "unhinted") and refname != "vs_unhinted" or g == "unhinted":
                    continue
                pg = (fg * nu + 0.5) / (nu + 1)
                pr = (rv * nref + 0.5) / (nref + 1)
                lo = logit(pg) - logit(pr)
                cand = np.nonzero((fg >= 0.05))[0]
                n_items_tok = {int(t): len({items[j] for j in np.nonzero(present_units[:, t])[0]}) for t in cand}
                cand = [t for t in cand if n_items_tok[int(t)] >= 3 and strings[t]]
                cand = sorted(cand, key=lambda t: -lo[t])[:15]
                rows = []
                for t in cand:
                    e50 = np.mean([U[k][1][t] for k in us]) / max(fg[t], 1e-9)
                    ea = np.mean([U[k][2][t] for k in us]) / max(fg[t], 1e-9)
                    rows.append({"tok": strings[t], "freq": round(float(fg[t]), 3), "ref": round(float(rv[t]), 3),
                                 "logodds": round(float(lo[t]), 2), "n_items": n_items_tok[int(t)],
                                 "n_items_total": len(set(items)), "echo50": round(float(e50), 2),
                                 "echo_any": round(float(ea), 2)})
                top_out[refname] = rows
            # most frequent tokens in this group (what shows up at all)
            freq_top = np.argsort(-fg)[:12]
            top_out["most_frequent"] = [{"tok": strings[t], "freq": round(float(fg[t]), 3)} for t in freq_top if strings[t]]
            # concepts and option tokens: unit means, item bootstrap
            conc = np.stack([U[k][3] for k in us])
            opt = np.stack([U[k][4] for k in us])  # [units, 2, 3]
            uniq_items = sorted(set(items))
            idx_by_item = defaultdict(list)
            for j, it in enumerate(items):
                idx_by_item[it].append(j)
            boots = []
            for _ in range(500):
                pick = rng.choice(len(uniq_items), len(uniq_items))
                jj = [j for p in pick for j in idx_by_item[uniq_items[p]]]
                boots.append(np.concatenate([conc[jj, :, 0].mean(0), opt[jj, :, 0].mean(0)]))
            boots = np.array(boots)
            point = np.concatenate([conc[:, :, 0].mean(0), opt[:, :, 0].mean(0)])
            names = list(CONCEPTS) + ["correct_option_word", "hinted_option_word"]
            top_out["concepts"] = {nm: {"share": round(float(point[i]), 4),
                                        "lo": round(float(np.percentile(boots[:, i], 2.5)), 4),
                                        "hi": round(float(np.percentile(boots[:, i], 97.5)), 4)}
                                   for i, nm in enumerate(names)}
            for i, nm in enumerate(CONCEPTS):
                sh = conc[:, i, 0].mean()
                top_out["concepts"][nm]["echo50"] = round(float(conc[:, i, 1].mean() / sh), 2) if sh > 0 else None
                top_out["concepts"][nm]["echo_any"] = round(float(conc[:, i, 2].mean() / sh), 2) if sh > 0 else None
            for i, nm in enumerate(("correct_option_word", "hinted_option_word")):
                s = opt[:, i, 0].mean()
                top_out["concepts"][nm]["echo50"] = round(float(opt[:, i, 1].mean() / s), 2) if s > 0 else None
                top_out["concepts"][nm]["echo_any"] = round(float(opt[:, i, 2].mean() / s), 2) if s > 0 else None
            top_out["n_units"] = nu
            top_out["n_items"] = len(uniq_items)
            cell[g] = top_out
        out["cells"]["|".join(ck)] = cell
        print("cell", ck, flush=True)
    json.dump(out, open(OUT / "lens_concepts_full.json", "w"), indent=1, ensure_ascii=False)
    print("saved")




# ---------------------------------------------------------------- stage C: lens strips
STRIP_ANCHORS = ["last_prompt", "cot_03", "cot_08", "cot_13", "think_end", "pre_answer"]


def stage_c(args) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")
    res = pickle.load(open(OUT / "stage_a.pkl", "rb"))
    by = {r["run_id"]: r for r in res}
    want = args.runs.split(",") if args.runs else []
    idx = {}
    for rdir in READOUTS:
        for line in open(R / rdir / "index.jsonl"):
            m = json.loads(line)
            idx.setdefault(m["run_id"], (rdir, m))
    out = {}
    for rid in want:
        rdir, m = idx[rid]
        with np.load(R / rdir / m["file"]) as z:
            top, toplp = z["top_ids"], z["top_lp"]
        names = m["top_anchor_names"]
        strip = {}
        for a in STRIP_ANCHORS:
            if m["named_positions"].get(a, -1) < 0:
                continue
            k = names.index(a)
            strip[a] = {L: [tok.decode([int(t)]).strip() or repr(tok.decode([int(t)])) for t in top[0, li, k, :5] if t >= 0]
                        for li, L in enumerate(LAYERS) if L >= 14}
        out[rid] = {"group": by.get(rid, {}).get("group"), "answer": m["answer"], "correct": m["correct_letter"],
                    "hint": m.get("hint_letter"), "strip": strip}
    json.dump(out, open(OUT / "strips.json", "w"), indent=1, ensure_ascii=False)
    for rid, d in out.items():
        print(f"\n#### {d['group']}: `{rid}` (correct {d['correct']}, hint {d['hint']}, answer {d['answer']})\n")
        ans = list(d["strip"])
        print("| Layer | " + " | ".join(ans) + " |")
        print("| --- |" + " --- |" * len(ans))
        for L in LAYERS:
            print(f"| {L} | " + " | ".join(" · ".join(x.replace("|", "\\|").replace("\n", "⏎") for x in d["strip"][a][L]) for a in ans) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "c"], required=True)
    ap.add_argument("--runs", default="")
    a = ap.parse_args()
    {"a": stage_a, "b": stage_b, "c": stage_c}[a.stage](a)
