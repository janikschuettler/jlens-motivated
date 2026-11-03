"""Per-item content-word tracked tokens (lens env, CPU; tokenizer only).

For each item and each option: the first content word of the option text that
no other option of the item contains (case-insensitive; content word =
alphabetic, >= 2 letters, not a stopword), and the first token of that word in
three forms: " word", "word", " Word" (as written in the option, bare, and
capitalised with a leading space). Each chosen ID must decode to a string
that encodes back to exactly that one ID (checked; the script stops
otherwise, rule 1). Duplicate IDs within an option (e.g. " Paris" = " Paris")
are kept once.

An item is excluded (and counted by reason) when any option has no
distinctive content word (e.g. numeric options), or when a token ID of one
option is also a token ID of another option (e.g. " photo" for "photography"
and "photosynthesis"), or when the " word" token has fewer than 2 characters
(" U" for "Ursa": too generic). A "word" or " Word" token under 2 characters
("C" for "Cygnus") is left out of that option's set instead.

Fixed before any readout (rule 3): the output, configs/content_tokens.json, is
committed; readout_runs.py --content-tokens re-checks every ID against the
tokenizer and refuses runs whose item is not in the file.

uv run --project envs/lens python scripts/lens/content_tokens.py \
    --items data/explore/gap_factorial/screen/items.jsonl --out configs/content_tokens.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from jmr import config

STOPWORDS = set("""
a an the of in on at to for from by with without into onto upon over under about above below between among
through during before after since until than then and or nor but so yet if because while whereas although though
as is are was were be been being am do does did done has have had having can could may might must shall should
will would not no none all any both each either neither every some such only own same other another more most
less least very too also just its it this that these those there here which who whom whose what when where why
how they them their he him his she her we us our you your i me my mine one ones two three four five six seven
eight nine ten first second third true false yes above below following none option options answer answers
only always never often sometimes usually etc vs per via
""".split())

# Chosen default: a first token shorter than 2 characters (" U", " c", "C") is
# too generic to read as an option. If the " word" token is that short, the item
# is excluded; a short "word" or " Word" token is left out of that option's set
# (counted in forms_dropped_too_short), since those forms are the less common
# ones in running text.
MIN_LETTERS = 2
WORD = re.compile(r"[A-Za-z]+")


def content_words(text: str) -> list[str]:
    return [w for w in WORD.findall(text) if len(w) >= 2 and w.lower() not in STOPWORDS]


def first_token(tok, text: str) -> int:
    return tok(text, add_special_tokens=False).input_ids[0]


def check_single(tok, tid: int) -> str:
    s = tok.decode([tid])
    enc = tok(s, add_special_tokens=False).input_ids
    if enc != [tid]:
        raise ValueError(f"token {tid} decodes to {s!r}, which encodes to {enc}, not one token (rule 1)")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mcfg = config.load("model")
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])

    items = {}
    for p in args.items:
        for line in open(p):
            d = json.loads(line)
            if d["id"] in items and items[d["id"]]["options"] != d["options"]:
                raise ValueError(f"item files disagree on {d['id']}")
            items.setdefault(d["id"], d)

    out, excluded = {}, {}
    reasons, dropped_forms = Counter(), Counter()
    n_whole = n_opt = 0
    for q, d in sorted(items.items()):
        words = [content_words(t) for t in d["options"]]
        lower = [{w.lower() for w in ws} for ws in words]
        chosen, why = {}, None
        for li, (L, ws) in enumerate(zip(d["letters"], words)):
            others = set().union(*(lower[j] for j in range(len(words)) if j != li))
            w = next((w for w in ws if w.lower() not in others), None)
            if w is None:
                why = "an option has no distinctive content word"
                break
            forms = {"space": " " + w, "bare": w, "cap": " " + w[0].upper() + w[1:]}
            ids = {f: first_token(tok, s) for f, s in forms.items()}
            labels = {f: check_single(tok, i) for f, i in ids.items()}
            if len(labels["space"].strip()) < MIN_LETTERS:
                why = f"the ' word' first token has < {MIN_LETTERS} characters"
                break
            short = [f for f in ("bare", "cap") if len(labels[f].strip()) < MIN_LETTERS]
            for f in short:
                dropped_forms[f] += 1
            keep = [f for f in forms if f not in short]
            whole = len(tok(forms["space"], add_special_tokens=False).input_ids) == 1
            chosen[L] = {"word": w, "ids": ids, "labels": labels, "whole_word_single_token": whole,
                         "forms_used": keep, "unique_ids": sorted({ids[f] for f in keep})}
        if why is None:
            seen = {}
            for L, c in chosen.items():
                for i in c["unique_ids"]:
                    if i in seen and seen[i] != L:
                        why = "options share a first token"
                    seen[i] = L
        if why is not None:
            reasons[why] += 1
            excluded[q] = why
            continue
        out[q] = {"correct": d["correct"], "letters": d["letters"], "options": chosen}
        n_opt += len(chosen)
        n_whole += sum(c["whole_word_single_token"] for c in chosen.values())

    res = {
        "description": "Per-item content-word tracked tokens; built by scripts/lens/content_tokens.py. "
                       "Per option: first distinctive content word, first token of ' word', 'word', ' Word'.",
        "tokenizer": {"hf_id": mcfg["hf_id"], "revision": mcfg["revision"]},
        "n_items": len(items), "n_included": len(out), "n_excluded": len(excluded),
        "excluded_by_reason": dict(reasons), "excluded": excluded,
        "options_space_form_whole_word": f"{n_whole}/{n_opt}",
        "forms_dropped_too_short": dict(dropped_forms),
        "items": out,
    }
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("items", "excluded")}, indent=1))
    for q in list(out)[:8]:
        print(q, {L: (c["word"], c["labels"]["space"]) for L, c in out[q]["options"].items()})


if __name__ == "__main__":
    main()
