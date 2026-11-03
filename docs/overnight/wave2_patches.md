# Wave 2: patches for other agents' files

**One patch is needed in the gap agent's files (found 23:15, below as 0)**; nothing else (`src/jmr/explore/{__init__,spec,run,analysis}.py`, `configs/explore/gap_*.yaml`, `blocks.yaml`, `scripts/explore/run_arms.py`). Wave 2 reuses `jmr.explore.spec` read-only and has its own runner (`jmr.explore.w2_run`, `scripts/explore/run_w2.py`).

## 0. Needed: `scripts/explore/analyze_gap.py` overwrites the correct letter in `known_items.jsonl`

`screen()` writes `pool.append({**row, **acc[q]})`, where `acc[q] = {"correct": {setting: count}, "n": ..., "acc": ...}`. So the Question field `correct` (the correct letter) is replaced by a dict of counts.

Anything that builds Questions from `data/explore/known_items.jsonl` gets `correct = {"A": 10, "B": 9}`:

- `jmr.explore.spec.load_items` with `source: jsonl`;
- wave 2's `known_pool`.

With that value, `hinted_letter` can pick the true answer as "B", and every outcome label is wrong.

This hit wave 2 part 1 too: my `w2_known.py` had copied the same format. The first E1+E3 chunk on mayor (820 runs, 22:45–23:14) is invalid and was moved to `data/explore/w2/_invalid/`. Since then, `w2_known.py` writes `n_correct`, and `jmr.explore.w2.question_from_row` raises if `correct` isn't one of the item's letters.

Patch (`scripts/explore/analyze_gap.py`, in `screen()`):

```python
-    acc: dict[str, dict] = {q: {"correct": {}, "n": {}, "acc": {}} for q in items}
+    acc: dict[str, dict] = {q: {"n_correct": {}, "n": {}, "acc": {}} for q in items}
@@
-            acc[q]["correct"][setting], acc[q]["n"][setting], acc[q]["acc"][setting] = k, n, k / n
+            acc[q]["n_correct"][setting], acc[q]["n"][setting], acc[q]["acc"][setting] = k, n, k / n
@@
-        hist = [sum(1 for q in acc if acc[q]["correct"].get(setting) == k) for k in range(11)]
+        hist = [sum(1 for q in acc if acc[q]["n_correct"].get(setting) == k) for k in range(11)]
```

Until that is applied and `known_items.jsonl` regenerated, use the wave-2 pool built by `scripts/explore/w2_known.py` (mayor: `data/explore/w2/known_pool.jsonl`; same items, same ≥ 9/10 rule).

## Notes


1. **Gap tool on wave-2 output.** `jmr.explore.spec.label_row` / `to_run_record` would raise on wave-2 `runs.jsonl` rows, because of the extra `w2` field. Use `jmr.explore.w2_labels.w2_label` (`labels.jsonl` is already written by `run_w2.py`). No patch is proposed.

2. **Lens side (optional, `src/jmr/lens/batch.py`, the lens agent's file).** For E2 C3 runs, the `<think>` token is the start of the prefill, so it sits in the **prompt**. `plan_run` currently keeps `think_start` only if it is in the completion. So for C3 runs:
   - `think_start = -1`;
   - the CoT span falls back to the non-thinking rule `(P, pre_answer + 1)`, which includes `</think>\n\n`.

   Readouts still work, but the CoT anchors are slightly off. Suggested patch, after `think_start, think_end = in_completion(...)` in `plan_run`:

   ```python
   # Prefilled runs (wave 2, E2 C3): "<think>" ends the prompt, "</think>" is generated.
   if think_start < 0 and think_end >= 0 and a.think_start is not None:
       think_start = a.think_start
   ```

   This doesn't change non-thinking runs: their `</think>` is in the prompt, so `think_end` stays -1. It also doesn't change multi-turn E4 runs, whose prompts contain no `<think>`, because the template drops earlier thinking.
