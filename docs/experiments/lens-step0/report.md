# First lens readout: M&B setup, step 0 (non-thinking)

2026-11-02, arena9-mayor (RTX 4090). Exploratory; no claims. Derived numbers: [../data/lens_step0.json](data/lens_step0.json).

## Data

- **Not octavia's step0 set.** Copying `data/explore/gap_factorial/step0/` from octavia was not permitted (tonight's brief for this agent excludes octavia). Instead the same spec, `configs/explore/gap_step0.yaml` (unchanged), was re-sampled on mayor: `run_arms.py configs/explore/gap_step0.yaml --out-suffix _mayor` → `data/explore/gap_factorial/step0_mayor/` (480 runs, 47 s). The run IDs differ, so the seeds and samples differ from octavia's. Same items, prompts, hints and decoding: M&B's format, non-thinking, M&B decoding (T 0.1, repetition penalty 1.15), 32 `mb_replication` items (MMLU-Redux), unhinted ×3 plus an expert hint on each of the 4 options ×3.
- Readout: `readout_runs.py --runs data/explore/gap_factorial/step0_mayor/runs.jsonl --out data/readouts/step0_mayor --validate 2`. 480 runs in 2.4 min. M0.4 passes (median |Δ| 0.00004, p99 0.24), 32,212/32,212 rank/top-25 overlaps exact, and the 2 validated runs are identical to the single-run readout.
- Analysis: `scripts/lens/analyze_step0.py` (numpy, CPU, 10 s).

**Groups.** Known items are those with all 3 unhinted answers correct: 24 of 32. On those items, hinted runs with a wrong hint (216): **followed** (final = hint) 109 runs / 22 items, **resisted** (final = correct) 103 / 21, other 4 (excluded). Switch rate on known items 50%, in line with octavia's 50–58%. **Unhinted** reference: 72 runs / 24 items.

**Measures.** Letters in form `any` (log-sum-exp of " X", "X", "(X" log-probs; rank = min over the forms). Roles: correct, hinted, other wrong (mean over the 2 wrong letters not hinted; for unhinted runs, the mean over all 3 wrong letters). Group means are means of item means, with 95% bootstrap CIs (2,000 resamples of items). Followed − resisted differences are within-item (19 items have both).

**Echo caveat.** The letters are in the prompt (options and hint), and the response names them. 92% of followed runs name the correct letter before answering (median at 45% of the response), as do 100% of resisted and unhinted runs (median ~70%). "Echo-free" below means response positions before the first time the response writes the correct letter (t ≤ m − 2, so the position that predicts the letter itself is excluded too). The final-answer position is never echo-free.

## (c) Position before the final answer letter: the only place the groups differ

Mean log-prob (nats) [95% CI]. "corr" = correct letter, "hint" = hinted letter, "ow" = other wrong letters.

**Logit lens**

| Layer | followed corr | followed hint | followed ow | resisted corr | resisted hint | resisted ow | unhinted corr | unhinted wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 | −9.2 [−9.7, −8.8] | −9.3 | −8.9 [−9.4, −8.4] | −8.3 | −7.9 | −8.7 | −8.3 | −8.4 |
| 24 | −5.6 [−6.3, −4.8] | −2.2 | −6.4 [−6.9, −6.0] | −0.5 | −8.2 [−9.1, −7.3] | −9.2 [−9.9, −8.5] | −1.0 | −7.3 |
| 26 | −4.2 [−4.9, −3.4] | −0.9 | −5.0 [−5.4, −4.7] | −0.4 | −6.1 [−7.1, −5.2] | −7.1 [−7.8, −6.4] | −0.4 | −6.2 |
| 28 | −3.6 [−4.5, −2.8] | −1.1 | −4.4 [−4.9, −3.9] | −0.3 | −5.5 [−6.8, −4.3] | −6.6 [−7.9, −5.3] | −0.4 | −6.0 |
| 30 | −8.2 [−9.0, −7.3] | −0.0 | −10.6 [−11.0, −10.2] | −0.1 | −9.3 [−10.0, −8.6] | −11.3 [−11.8, −10.8] | −0.0 | −10.6 |
| 32 | −12.9 [−14.0, −11.5] | −0.0 | −15.7 [−16.3, −15.2] | −0.3 | −16.0 [−17.1, −14.8] | −17.4 [−18.4, −16.5] | −0.1 | −16.2 |

**J-lens**

| Layer | followed corr | followed hint | followed ow | resisted corr | resisted hint | resisted ow | unhinted corr | unhinted wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 | −26.7 | −26.2 | −26.4 | −24.0 | −23.8 | −23.8 | −20.8 | −20.5 |
| 24 | −27.4 [−30.1, −25.1] | −23.3 | −28.3 [−30.7, −26.3] | −19.8 | −27.4 [−29.1, −25.7] | −27.7 | −19.5 | −24.5 |
| 26 | −35.6 [−37.2, −33.9] | −31.2 | −36.3 [−37.5, −35.1] | −29.4 | −35.6 | −35.8 | −27.8 | −32.4 |
| 28 | −27.4 [−29.0, −25.7] | −22.4 | −28.4 [−29.5, −27.4] | −19.1 | −27.0 | −27.0 | −19.4 | −25.5 |
| 30 | −20.0 [−21.9, −17.6] | −8.0 | −22.8 [−23.8, −21.9] | −8.6 | −21.0 [−22.5, −19.8] | −22.5 [−23.5, −21.5] | −11.1 | −23.1 |
| 32 | −18.7 [−20.7, −16.2] | −0.9 | −22.1 [−23.0, −21.2] | −2.0 | −22.1 [−23.7, −20.6] | −23.9 [−25.3, −22.5] | −4.2 | −25.0 |

Layers 14–20 show no separation between any letters in either lens (all CIs overlap; full tables in the JSON). Mean log10 rank at layer 30, followed correct vs other wrong: logit 0.5 vs 0.8, J-lens 2.1 vs 2.5. At layer 32: logit 0.8 vs 1.2, J-lens 1.8 vs 2.2.

**What this shows, descriptively:**

- In **followed** runs, just before the model writes the hinted letter, the correct letter is above the other wrong letters at layers 24–32 in both lenses. The gap is about 0.8 nats at layers 24–28 (logit lens; the group CIs overlap) and 2.4–3.4 nats at layers 30–32 (logit lens and J-lens; the CIs don't overlap).
- But the same holds in mirror image for **resisted** runs: the unchosen *hinted* letter is above the other wrong letters, by about 1.0–2.0 nats at layers 24–32 (logit lens). An unchosen option the prompt and the response made salient keeps some probability. That is the obvious alternative to "the known answer is still represented", and these numbers don't separate the two. A fair comparison needs the correct letter in followed runs against a matched salient-but-wrong letter, or responses that never name the correct letter.
- Within items (followed − resisted, 19 items): the correct letter is 3.1–4.9 nats lower at layers 24–28 and 7.8–12.3 nats lower at layers 30–32 (logit lens). The J-lens differences are larger (6.2–16.2 nats). So the correct letter is pushed down a lot at the answer position when the model follows, not kept at resisted-run levels.

## (a) Last prompt token and (b) along the response: no differences

- **(a) last prompt token:** all letters and groups are within about 1 nat of each other at every layer and in both lenses, with overlapping CIs (e.g. logit lens layer 24: followed correct −14.7, resisted correct −14.7, unhinted correct −15.0). Followed and resisted samples of the same prompt share this position (same IDs), so a difference here could only come from item and hint composition.
- **(b) 20 bins along the response:** at layer 30 the correct, hinted and other-wrong letters sit together (logit lens about −30 → −26, J-lens about −43 → −36) in every group. The differences between roles are ≤ 0.5 nats in all bins, with all positions or with echo-free positions only. The mean over echo-free positions shows no group or role difference beyond the CIs at any layer, in either lens.

Taken together, the letters are only "read out" at the answer position. Averaging over the response dilutes any letter signal into positions where no letter is in play. This set is too short and too letter-dense to say anything about the correct answer being represented while the model argues for B.

## What I'd read out next

1. **Octavia's actual step0 set and the gap factorial** (thinking arms included): same commands, about 200 runs/min for non-thinking runs and about 85/min for thinking runs on a 4090. Thinking runs have a long reasoning span before any letter, which is where (b) can mean something.
2. **A salience-matched control at the answer position:** compare the correct letter in followed runs with the hinted letter in resisted runs (both "the unchosen salient option"), and with a wrong option the response discussed. This needs a per-run "letters mentioned" mask; the token IDs make it cheap.
3. **Positions just before the first mention of a letter** (anchors at m − 1 for each letter) instead of averages over the response, and the stored residuals at `pre_answer` for a simple linear probe (correct letter vs the others, cross-validated by item).
4. Wave-2 run sets with followed runs on reliably known items in thinking mode (the M2 contrast); temp_check has none.
