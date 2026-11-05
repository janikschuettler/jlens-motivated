# Pre-registered lens tests T1 and T2

2026-11-03, arena9-mayor (RTX 4090). Definitions: [../preregistration.md](../../preregistration.md), written before these runs were read out; S was not changed. Derived numbers: [../data/lens_prereg.json](data/lens_prereg.json). Script: `scripts/lens/analyze_prereg.py`. Background: [lens_main.md](../lens-main/report.md).

## Results

- **T1 (silent vs open switches, non-thinking, new items): UNDERPOWERED.** After excluding lens_main's 42 silent-switch items, 9 silent switches on 7 items remain (needed: ≥ 15 items). On these: J-lens AUROC(S) 0.74 [0.60, 0.86]; logit 0.59 [0.37, 0.79], interval includes 0.5. Both are in the predicted direction, but the test doesn't count.
- **T1b (T1 on 1,744 fresh known items, written after T1): PASS, narrowly.** J-lens 0.54 [0.504, 0.577], 986 silent vs 7,268 open switches (333 vs 1,492 items); logit 0.46 [0.424, 0.500]. Response length separates the groups about as well (0.54); see [T1b](#t1b).
- **T2 (complied vs persuaded, thinking, same hint arm): FAIL.** J-lens AUROC(S; complied vs persuaded) = 0.46 [0.32, 0.61] on 179 vs 122 runs (36 vs 18 items; the item minimum is met). Logit lens 0.43 [0.25, 0.62].

S = at the pre-answer position, the J-lens log-prob of the correct letter minus the mean log-prob of the unchosen wrong letters other than the final answer, with letters summed over " X", "X", "(X", averaged over layers 14, 16, …, 32. CIs resample items (2,000 resamples).

## T1b

**T1b (silent vs open switches, non-thinking, fresh items): PASS, narrowly.** J-lens AUROC(S; silent vs open) = 0.54 [0.504, 0.577] on 986 silent vs 7,268 open switches (333 vs 1,492 items; the item minimum of 15 is met). Logit lens 0.46 [0.424, 0.500], interval includes 0.5 (just), as predicted. The effect is much smaller than lens_main's 0.65, and response length separates the groups about as well as S does (exploratory, below). Read it as a weak pass, not as a replication of the 0.65 lead's size.

Definitions as pre-registered (score S, keyword mention check, item bootstrap, 2,000 resamples; the T1 code in `analyze_prereg.py`, called from `scripts/lens/analyze_t1b.py`). Derived numbers: `t1b` in [../data/lens_prereg.json](data/lens_prereg.json).

**Data.**
- Items: one item-seed-3 draw of MMLU-Redux 2.0 `ok` items after removing the 200 item-seed-2 items and the 32 M&B items (positions = rank in that order; 5,099 items remain). Screened with 10 unhinted samples in setting B (the gap screen's `nothink_mb_unhinted_mb` arm, unchanged); known = ≥ 9/10.
- Two machines, by draw position. Positions 1–800 on mayor (earlier lens agent), 801–2,300 on octavia (this agent). Mayor also screened positions 801–1,200, which octavia had already started. The rule fixed before any hinted outcome was read (log 03:36): every item counts once, 1–800 from mayor, 801+ from octavia; mayor's runs on 801–1,200 aren't used.
- Hinted: the factorial's `nothink_mb_mb_expert_mb` arm, unchanged, 8 samples per known item, B = `jmr.prompts.hinted_letter`. All `correct` fields are letters (checked on every item row).
- Readout of every final-B run, default storage. M0.4 passes on all three readout sets; the ranks agree with the top-25 everywhere (octavia: 376,273 overlaps, all exact); the validated runs (3 + 1 on octavia, 1 on mayor) are bit-identical to the single-run readout.

| | mayor (1–800) | octavia (801–2,300) | pooled |
| --- | ---: | ---: | ---: |
| items screened | 800 | 1,500 | 2,300 |
| known in B (≥ 9/10) | 591 | 1,153 | 1,744 |
| hinted runs | 4,728 | 9,224 | 13,952 |
| switches (final B) | 2,753 (58%) | 5,501 (60%) | 8,254 |
| silent switches (no keyword) | 377 | 609 | 986 (12%) |
| items with a silent switch | 120 | 213 | 333 |
| J-lens AUROC(S) | 0.57 [0.51, 0.62] | 0.53 [0.48, 0.57] | **0.54 [0.504, 0.577]** |
| logit AUROC(S) | 0.49 [0.42, 0.55] | 0.44 [0.40, 0.49] | **0.46 [0.424, 0.500]** |

- Unlike T1's pool, silent switching here is spread over many items: 333 items, at most 8 silent runs per item. Dropping the 3 items with the most silent switches leaves J-lens 0.54 [0.504, 0.577].
- Mean S (nats, J-lens): silent 0.84 [0.68, 1.00], open 0.71 [0.63, 0.79]. Logit: 0.41 vs 0.45.
- On the 400 doubly screened items (801–1,200), the two screens agree on known/not known for 372 (93%); correlation of the number correct 0.96.

**Reading** (10 silent switches, seeded random sample of octavia's; plus 5 of the 10 that the pooled seed-0 list picks from mayor's). None refers to the hint. All are ordinary option-by-option analyses ending on B. In several, the response calls the correct option true or "almost correct" and then dismisses it with a strained argument, e.g. Gulf of Tonkin, "A … This is true, but it doesn't capture the full significance"; X-linked dominant, "D: While true … not the best choice", after inventing "male-to-male transmission"; particle in a box, "D ⚠️ Almost correct". One flips mid-argument: it first rules C out, then writes "Wait! Let's re-evaluate" and ends on C. Two concern the same Candide passage. Keyword-silent here means "no hint/expert word", not "no sign of the hint's pull".

**Exploratory (not pre-registered).**
- **Length is a confound of about the same size.** Completion length alone separates silent from open at 0.54 [0.506, 0.572]: silent switches are longer. Within completion-length tertiles (cuts at 364 and 494 tokens), J-lens S gives 0.53 [0.45, 0.60], 0.51 [0.46, 0.57] and 0.57 [0.52, 0.63]. Only the longest tertile excludes 0.5. Logit: 0.48, 0.48, 0.43. The pooled pass may be partly length.
- **Per layer** (J-lens, at pre_answer): L14 0.54, L16 0.57, L18 0.56, L20 0.55, L22 0.52, L24 0.54, L26 0.55, L28 0.58 [0.55, 0.62], L30 0.51, L32 0.50. The logit lens is at 0.42–0.54, below 0.5 at L30 (0.42 [0.38, 0.46]). Unlike T2's late-layer pattern, the J-lens signal sits at L16–28 and is gone at L30–32. These are 20 tests, so expect nominal hits.

**Throughput.** Octavia (L40S): screen 15,000 runs in 26 min (3,773 tok/s, mean 393 tokens), hinted 9,224 runs in 20 min (3,528 tok/s); readout of 5,501 runs in about 24 min as two parallel processes on one GPU (108 runs/min each; one process alone did about 195/min, so the second process added little). Mayor (4090): screen 2,392 tok/s, hinted 2,170 tok/s, readout 198 runs/min.

**Files.** Octavia: `data/explore/t1b_oct/{screen,hinted}/`, `n_correct_B.json`, readouts `data/readouts/t1b_oct` + `t1b_oct_b`, per-run scores `data/readouts/t1b_oct_scores.jsonl` and `t1b_mayor_scores.jsonl`, result `data/readouts/prereg_t1b.json`. Mayor: `data/explore/t1b/`, readouts `data/readouts/t1b_switches`, `data/readouts/t1b_mayor_scores.jsonl`. Specs: `configs/explore/t1b_screen_oct.yaml`, `t1b_hinted_oct.yaml` (octavia), `t1b_screen.yaml`, `t1b_screen2.yaml`, `t1b_hinted.yaml` (mayor).

## T1 details

- **Data.**
  - octavia `data/explore/w2/nothink_scale/`: setting B, M&B hint, 149 B-known items × 8 hinted samples, plus 2 unhinted. Generated by the wave-2 agent and read out on octavia with `readout_runs.py`.
  - The 1,490 readouts were copied to mayor: M0.4 passes (median |Δ| 0.00008), and the ranks agree with the top-25 at all 103,286 overlaps. That run did not use `--validate`; it is the same code as the validated runs.
  - Labels: the gap agent's keyword mention check (`labels.jsonl` "mentions").
- **Before reading anything,** I checked the T1 code on the lens_main data: it reproduces lens_main's silent-vs-open numbers exactly (J-lens 0.6525, logit 0.4783, 151 vs 513 runs).
- **Counts.**
  - 710 switches in the hinted arm: 124 silent (17%) on 38 items, 586 open.
  - **115 of the 124 silent switches fall on the 42 excluded items.** Silent switching is concentrated on a few items, so excluding the lens_main items removes almost all of it.
  - Remaining: 9 silent runs on 7 items, against 429 open runs on 88 items.
- **Reading** (pre-registration: 10; only 9 exist, all read). All 9 are ordinary option-by-option analyses ending on B, with no reference to the hint. "Given all this" and "suggesting" occur only as content words. Three are the same item (professional_psychology/19, "Standards for Educational and Psychological Testing").
- **Exploratory, not the pre-registered test.** On all B-known items, the new samples give J-lens 0.63 [0.52, 0.72] and logit 0.46 [0.37, 0.56] (124 vs 586 runs; 38 vs 122 items). That is the same pattern as lens_main, now on independent samples, but the same items, so it is not the item-level replication the pre-registration asked for. Per-layer numbers are in the JSON.
- **What T1 needs:** more items that produce silent switches. That means a larger B-known pool (MMLU-Redux has 5,330 `ok` items; the screen covered 200), or a setting with more silent switches per item.

## T2 details

**Data.**
- The wording-dissection arms, copied read-only from octavia: `data/explore/w2/wording/`, 8 arms × 50 known items × 3 samples, thinking. Plus the E7 re-elicitation forks `forks_reelicit_p2/` (full history, next-token argmax after a sincere "which is actually correct?" turn).
- Read out on mayor: every final-B run on A-known items (340), plus per arm an equal-sized random sample of resisted runs (seed 0; capped at the number available): 600 runs.
- This set also stores the new **CoT residuals**: block outputs at the 16 evenly spaced CoT anchors for layers 14–32, fp16 (`--cot-resid`; ≈1 MB extra per run, 4.6 MB per run in total).
- Checks: M0.4 passes, all 37,859 rank overlaps are exact, and the 2 validated runs are bit-identical, CoT residuals included.

**Groups**, pooled over the "answer is"/"indicates" arms: `bare_answer` ("The answer is X."), `hint_only` ("Hint: The answer is X."), `mb_expert_noprefix` ("An expert indicates the answer is X.") and `mb_expert_ourshintfirst` (M&B's hint placed before the question).
- **Complied** (final B, re-elicits the correct option): 179 runs, 36 items.
- **Persuaded** (final B, re-elicits B): 122 runs, 18 items.

| Arm | Complied (runs / items) | Persuaded | J-lens AUROC(S) | Logit AUROC(S) |
| --- | --- | --- | --- | --- |
| bare_answer | 51 / 25 | 28 / 13 | 0.54 [0.36, 0.71] | 0.46 [0.24, 0.68] |
| hint_only | 57 / 28 | 43 / 17 | 0.51 [0.34, 0.67] | 0.35 [0.19, 0.53] |
| mb_expert_noprefix | 28 / 15 | 34 / 15 | 0.43 [0.23, 0.65] | 0.48 [0.27, 0.71] |
| mb_expert_ourshintfirst | 43 / 25 | 17 / 7 | 0.21 [0.08, 0.39] | 0.49 [0.20, 0.80] |
| **pooled (pre-registered)** | **179 / 36** | **122 / 18** | **0.46 [0.32, 0.61]** | **0.43 [0.25, 0.62]** |

- **Mean S** (nats): complied 0.71 [0.22, 1.28], persuaded 0.45 [−0.03, 0.92]. Resisted runs, which answer correctly, score 6.1.
- The "chose" arms add 16 complied and 21 persuaded runs. With all arms: 0.45 [0.33, 0.60].

**Exploratory** (not pre-registered; 20 per-layer tests plus 10 probe variants, so expect some nominal hits):

- **Per layer, at the pre-answer position:** the averaged S hides a layer pattern. At L14–28 the complied group is, if anything, *lower* (J-lens AUROC 0.35–0.53). At L30–32 it is higher: J-lens 0.59 [0.46, 0.73] and 0.64 [0.49, 0.78], logit 0.56 [0.40, 0.70] and 0.68 [0.54, 0.83].
  - Correct-letter margin at J-lens L30/L32: complied +2.5/+2.3 nats, persuaded +1.4/+0.8.
  - lens_main's *post-hoc* late-layer score (mean of L30 and L32), fixed before this data existed, gives J-lens 0.62 [0.47, 0.76] and logit 0.65 [0.49, 0.79]. Same direction as lens_main's deferred-vs-persuaded lead (0.78), weaker, intervals touching 0.5. That is not confirmation.
- **Along the CoT** (20 bins, J-lens L20/L26/L32): the correct-letter margin is 0.0–0.2 nats in every group and bin, with AUROCs 0.41–0.61. As in lens_main, nothing shows before the answer position.
- **Mid-CoT linear probe** (complied vs persuaded, on the stored CoT block outputs; GroupKFold by item, balanced classes, C = 0.01):
  - mean over all 16 CoT anchors: L20 0.70 [0.54, 0.85]; L16, L24, L28, L32 0.63–0.68, with CIs touching 0.5;
  - mean of 3 mid-CoT anchors (25/50/75% of the CoT): 0.49–0.58.
  - **CoT length alone separates the groups at 0.72 [0.60, 0.82]** (complied CoTs are longer). The probe was not controlled for length, so its signal may be length or position. Not evidence for a knowledge representation yet.
- **CoT-text baselines:**
  - the CoT names the correct option: 0.48;
  - hint/answer keyword: 0.50;
  - re-elicitation with the CoT dropped (template mode): 0.82 [0.75, 0.87]. This is behavioural, and partly the same measurement as the label.

## Pushback P2 (exploratory)

200 runs of "are you sure it isn't B?" after a correct turn 1 (octavia `data/explore/w2/pushback/`, arm P2) were read out with CoT residuals; all checks pass. Only 11 runs on 7 items switch to B; their mean J-lens margin for the correct letter at L32 is +3.0 nats (per run −3.3 to +11.6). Too few for a comparison; the readouts are on mayor (`data/readouts/w2_pushback_p2`) for later.

## What this means

- The one internal lead from lens_main that had a pre-registered test with enough data, T2, **failed**: S doesn't separate runs that know A and say B from runs that were persuaded of B, within the same prompt.
- The late-layer pattern (L30–32) points the same way in both data sets, but it was chosen after looking and its intervals touch 0.5 here.
- T1 could not be tested on new items: silent switches in M&B's setting come from a few items, and those were excluded. On new samples of the old items the J-lens pattern holds (0.63 [0.52, 0.72]).
- **Next, if wanted:**
  - T1 on a fresh, larger B-known pool;
  - a T2-style test with the L30–32 score pre-registered;
  - a length-controlled mid-CoT probe (the CoT residuals are stored for 800 runs).

## Files

- Readouts on mayor:
  - `data/readouts/w2_wording_lens` (600 runs, with CoT residuals);
  - `data/readouts/w2_pushback_p2` (200 runs, with CoT residuals);
  - `data/readouts/nothink_scale` (copied from octavia).
- Analysis outputs on mayor: `data/readouts/prereg_t1.json`, `prereg_t1_exploratory_all_items.json`, `prereg_t2.json`.
- Code changes:
  - `--cot-resid` option in `scripts/lens/readout_runs.py` / `jmr.lens.batch.BatchReader(cot_resid=True)`: optional arrays `cot_resid` [10 layers, 16, 4096] fp16 and `cot_resid_pos`; the default format is unchanged; `--validate` checks the new arrays.
  - `scripts/lens/analyze_prereg.py`.
