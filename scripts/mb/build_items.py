"""Draw the M&B-replication item set: raw cais/mmlu test rows that also appear in
MMLU-Redux 2.0 (matched by subject + question text + choices), 200 at random.

Run with the third-party env (it has `datasets`):
  third_party/motivated-reasoning/.venv/bin/python -I scripts/mb/build_items.py
Writes data/runs/mb_replication/items.jsonl and items_ds/ (a datasets.Dataset on disk,
read by the patched get_dataset via MB_LOCAL_DATASET).
"""
import collections
import json
import os
import random
import sys

from datasets import Dataset, get_dataset_config_names, load_dataset

N_ITEMS = 200
SEED = 0
OUT = "/root/jlens-motivated/data/runs/mb_replication"
REDUX = "edinburgh-dawg/mmlu-redux-2.0"


def redux_label(row):
    """Redux-corrected label index (None when Redux gives no usable correction)."""
    if row["error_type"] == "ok":
        return row["answer"]
    ca = row["correct_answer"]
    if ca is None:
        return None
    ca = str(ca).strip()
    if ca.isdigit() and int(ca) < len(row["choices"]):
        return int(ca)
    if len(ca) == 1 and ca.upper() in "ABCD":
        return "ABCD".index(ca.upper())
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    index = collections.defaultdict(list)
    for i, (q, s, c) in enumerate(zip(mmlu["question"], mmlu["subject"], mmlu["choices"])):
        index[(s, q, tuple(c))].append(i)

    matched, n_redux, n_zero, n_multi, ans_mismatch = [], 0, 0, 0, 0
    error_types = collections.Counter()
    for cfg in get_dataset_config_names(REDUX):
        red = load_dataset(REDUX, cfg, split="test")
        for j, row in enumerate(red):
            n_redux += 1
            error_types[row["error_type"]] += 1
            hits = index.get((cfg, row["question"], tuple(row["choices"])), [])
            if len(hits) == 0:
                n_zero += 1
                continue
            if len(hits) > 1:
                n_multi += 1
                continue
            m = mmlu[hits[0]]
            if m["answer"] != row["answer"]:
                ans_mismatch += 1
            matched.append({
                "item_id": f"redux/{cfg}/{j}",
                "mmlu_index": hits[0],
                "subject": cfg,
                "question": m["question"],
                "choices": m["choices"],
                "answer": m["answer"],
                "redux_answer": row["answer"],
                "redux_error_type": row["error_type"],
                "redux_correct_answer_raw": None if row["correct_answer"] is None else str(row["correct_answer"]),
                "redux_label": redux_label(row),
            })
    stats = {
        "n_redux_rows": n_redux, "n_matched_unique": len(matched), "n_no_match": n_zero,
        "n_multi_match": n_multi, "n_answer_mismatch_mmlu_vs_redux": ans_mismatch,
        "redux_error_types_all": dict(error_types), "seed": SEED, "n_items": N_ITEMS,
    }
    rng = random.Random(SEED)
    drawn = rng.sample(matched, N_ITEMS)
    stats["drawn_error_types"] = dict(collections.Counter(d["redux_error_type"] for d in drawn))
    stats["drawn_redux_label_none"] = sum(d["redux_label"] is None for d in drawn)
    with open(os.path.join(OUT, "items.jsonl"), "w") as f:
        for d in drawn:
            f.write(json.dumps(d) + "\n")
    with open(os.path.join(OUT, "items_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    cols = {k: [d[k] for d in drawn] for k in ["question", "subject", "choices", "answer", "item_id", "mmlu_index"]}
    Dataset.from_dict(cols).save_to_disk(os.path.join(OUT, "items_ds"))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    sys.exit(main())
