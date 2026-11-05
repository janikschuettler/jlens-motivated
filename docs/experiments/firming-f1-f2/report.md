# F1 and F2: firming the J-lens nulls and the planted-claim positive

Pre-registered (docs/preregistration.md, F1-F3; not blind: the point estimates were already known). CPU only, on stored data. F3 (code review) not part of this report. All intervals resample questions (5,000 resamples for F1, seed 0, same resamples for both lenses; 2,000 for F2).

## Answers

**F2: does the planted-claim 0.71 survive a question-difficulty control? Yes, as a run-level signal, by the pre-registered rule.** Compared within the same question, the J-lens score still separates runs that name the correct answer when re-asked from runs that don't (0.86 [0.72, 0.95], permutation p = 0.0001). After adjusting for difficulty, the score still carries a positive effect (log-odds per SD 0.57 [0.12, 1.43]). But difficulty alone predicts the outcome about as well as the score does (0.72), only 11 questions have both outcomes, and the logit lens does nearly as well within question (0.72), so the J-lens advantage over the logit lens is still not shown.

**F1: are the main J-lens nulls equivalent to chance (within +-0.10)? Only some.** Five of 17 case analyses are equivalent to chance at every position tested. Most others are inconclusive: with 18-60 questions per group the intervals are too wide to rule out an effect of 0.10. The J-lens advantage over the logit lens is ruled out (below +0.10) in about half of the cases, but not for the cases where it matters most (silent vs open switches at the answer, T1b, the planted claim).

## F2 results (389 kept-claim runs, 74 questions, 199 name the correct answer when re-asked)

| What was checked | J-lens | Logit lens | How to read it |
| --- | --- | --- | --- |
| All runs, no control (original test) | 0.71 [0.57, 0.84] | 0.60 [0.47, 0.73] | Reproduces R-C3. |
| Difficulty alone (unhinted misses out of 20; easier = names correct) | 0.72 [0.62, 0.82] | | A question's difficulty predicts the outcome as well as the score. |
| Control run wrong alone (fresh-context control right = names correct) | 0.73 [0.62, 0.83] | | Same. |
| Compare only runs of equal difficulty (difficulty bands x control right/wrong) | 0.73 [0.59, 0.85] | 0.60 [0.43, 0.79] | The score still separates once difficulty is held equal. |
| Only the easiest questions (no unhinted misses, control right; 257 runs, 49 questions) | 0.73 [0.59, 0.86] | 0.61 [0.43, 0.80] | Same, on the best-known questions. |
| Same question only (11 questions with both outcomes, 90 runs) | 0.86 [0.72, 0.95] | 0.72 [0.59, 0.87] | Permutation p = 0.0001 (J-lens), 0.0013 (logit). 9 of 11 questions go the expected way, 1 tied. |
| J-lens minus logit lens, same question | +0.13 [-0.04, +0.28] | | Not shown to differ. |
| Adjusted J-lens effect (log-odds per SD of the score, adjusted for difficulty and control) | 0.57 [0.12, 1.43] | 0.39 [-0.16, 1.32] | Interval excludes 0 for the J-lens, not for the logit lens. Without adjustment: 0.74 [0.21, 1.57]. |
| Adjusted effect, secondary covariates added (thinking-off misses, control p(correct), recovery rate) | 1.05 [0.36, 2.88] | 0.19 [-0.38, 1.52] | Same direction. |

How to read: AUROC 0.5 is chance, 1 is perfect. "Matched" AUROC counts only pairs of runs from the same difficulty band, so it cannot be driven by which questions are hard.

Decision by the pre-registered rule: within-question p < 0.05 and the adjusted J-lens interval excludes 0, so **the 0.71 survives as a run-level signal**.

Interpretation line: the planted-claim lead is not just question difficulty. Confidence moderate: 11 questions give the within-question result, the adjusted interval is wide and its lower end is close to 0, and the result is one case in one model. It does not change the earlier statement that the J-lens is not shown to beat the logit lens (J minus logit +0.13, interval includes 0).

Sensitivity (not pre-registered): the fresh-context control puts less than half its probability on the answer letters for 21% of runs (the model wants to reason first; for 19 forks the letter probabilities underflow to exactly 0). Dropping those runs (309 runs, 55 questions) gives matched AUROC 0.69 [0.54, 0.84], within-question 0.81 [0.62, 0.95] (p = 0.0002, 8 questions) but the adjusted coefficient 0.45 [-0.03, 1.37] includes 0. So the "survives as a run-level signal" verdict relies on those runs; the within-question result does not.

## F1 results (J-lens against chance; equivalence bound AUROC 0.40-0.60)

Equivalent = the 90% interval lies inside 0.40-0.60. Case verdict "equivalent" needs every position equivalent. "Signal" = a 95% interval excludes 0.5 (in either direction; direction noted). Score: correct-minus-pushed letter (cases 1), correct-minus-uninvolved letters (T1b, T2, R-C3, R-CL), hack-word score (hack), eval-word score (EA27). J-lens minus logit lens: "J advantage >= 0.10 ruled out" = upper end of the 90% interval below +0.10.

| Case (positive vs negative) | Position(s) | J-lens AUROC, 90% interval (per position) | Case verdict, J-lens vs chance | J-lens advantage >= 0.10 ruled out? |
| --- | --- | --- | --- | --- |
| Told to argue for B vs no hint (348 vs 80) | last prompt / mid / end of reasoning | 0.54 [0.50, 0.59] / 0.42 [0.35, 0.47] / 0.47 [0.45, 0.49] | Signal, but below 0.5 (mid, end); not equivalent at mid | Yes, all |
| Hint followers vs resisters (340 vs 260) | same | 0.59 [0.49, 0.68] / 0.50 [0.42, 0.58] / 0.58 [0.48, 0.67] | Inconclusive | Yes, all (logit lens is ahead at mid: J minus logit -0.11 [-0.20, -0.02]) |
| Silent vs open switches, non-thinking (609 vs 4,892) | same + answer | 0.49 / 0.54 / (no end) / 0.54 [0.50, 0.58] | **Equivalent to chance** (no 95% interval excludes 0.5) | **No** at the answer: J minus logit +0.11 [+0.06, +0.16] (logit lens is below chance, 0.43) |
| Kept planted claim vs recovered (61 vs 134, 21 vs 33 questions) | same | 0.58 [0.44, 0.72] / 0.50 [0.38, 0.62] / 0.46 [0.34, 0.59] | Inconclusive | Yes, all |
| Grader-bug followers vs resisters, Qwen3-14B thinking (163 vs 230) | same | 0.52 / 0.49 / 0.55 [0.48, 0.62] | Inconclusive (end of reasoning not equivalent) | No (J minus logit +0.11 [+0.03, +0.19] at end of reasoning) |
| same, gpt-oss-20b (603 vs 241) | same | 0.49 / 0.47 / 0.46 [0.43, 0.49] | **Equivalent to chance** | Yes |
| same, Qwen3-8B no-think (189 vs 189) | last prompt, mid | 0.52 / 0.53 | **Equivalent to chance** | Yes |
| same, Qwen3-14B no-think (692 vs 261) | last prompt, mid | 0.54 / 0.56 [0.52, 0.61] | Signal at mid (nonzero, small); mid not equivalent (upper end 0.606) | Yes |
| T1b pre-registered, silent vs open switches, answer position, pooled (986 vs 7,268) | answer | 0.542 [0.511, 0.573] | **Equivalent to chance** (and a small nonzero effect: 95% interval 0.504-0.577) | **No**: J minus logit +0.08 [+0.04, +0.12] |
| T1b octavia half (609 vs 4,892) / mayor half (377 vs 2,376) | answer | 0.527 [0.488, 0.567] / 0.565 [0.518, 0.610] | Equivalent (octavia half) / signal, not equivalent (mayor half) | No / No |
| T2, followers who still know vs persuaded (179 vs 122; 42 questions) | answer | 0.456 [0.338, 0.581] | Inconclusive | No (J minus logit +0.02 [-0.20, +0.25]) |
| Claimed-answer-key repeat R-CL, deferred vs persuaded (40 vs 55) | answer | 0.539 [0.395, 0.673] | Inconclusive | No (-0.01 [-0.12, +0.10]) |
| Reward-hack model, hidden hack vs no hack, problem means (59 problems) | mid / last prompt / end of reasoning / before code | 0.44 [0.35, 0.53] / 0.51 [0.49, 0.53] / 0.57 [0.48, 0.67] / 0.59 [0.51, 0.67] | Inconclusive (only last prompt equivalent) | Yes at mid, last prompt, before code; no at end of reasoning |
| Reward-hack, within-problem pairs, mid (secondary) | mid | 0.447 [0.352, 0.542] | Inconclusive | Yes |
| Qwen3.6-27B, "testing me" vs not, before the first such sentence (74 vs 40; 19 questions) | before first sentence | 0.485 [0.392, 0.582] | Inconclusive (19 questions: intervals likely too narrow) | Yes |
| Planted-claim repeat R-C3 (secondary, positive for scale; 199 vs 190) | answer | 0.709 [0.591, 0.819] | Signal | No (J minus logit +0.11 [-0.07, +0.27]) |

How to read the table: an "equivalent" verdict means we can rule out a J-lens effect larger than 0.10 AUROC in either direction. "Inconclusive" means the data fit both chance and a moderate effect. Full per-position numbers, 95% intervals, logit lens, smallest bound each cell would meet, and the J-minus-logit differences: `data/f1_all.json`.

Interpretation lines:
- The nulls that are firm (equivalent to chance) are silent vs open switches (non-thinking), the GPT-OSS-20B and Qwen3-8B no-think grader-bug cases, and the T1b answer-position test. For these we can say the J-lens effect is smaller than 0.10 AUROC.
- The nulls on the planted claim before the answer, on followers vs resisters, on the hack model (mid reasoning, the pre-registered cell), on T2, on R-CL and on Qwen3.6-27B eval awareness are inconclusive. Say "not shown", not "absent".
- One finding contradicts a current summary: at the answer position in T1b (silent vs open switches) the J-lens is ahead of the logit lens by 0.08 [0.04, 0.12] (pooled), because the logit lens is slightly below chance (0.46). The advantage is small and both lenses are close to chance, but "the J-lens never beats the logit lens in paired tests" (docs/findings.md, claim 7) is not exactly right for this case; the same holds for silent vs open at the answer in the competitors set (+0.11 [+0.06, +0.16]) and for Qwen3-14B grader-bug end of reasoning (+0.11 [+0.03, +0.19]). The logit lens is ahead in two cases (wording mid, -0.11; gpt-oss mid and end, -0.04 to -0.05).
- Confidence moderate. The intervals resample questions, which is right for the clustering but is likely too narrow for cases with about 20 clusters (planted claim: 21 vs 33 questions; EA27: 19 questions).

## Deviations and notes (none loosens a threshold)

1. **Control with no letter probability.** In 19 control forks (of 239) all letter probabilities underflow to 0 and the stored argmax is None. The pre-registered rule "control argmax is not the correct letter" was applied literally (these count as "control wrong"); their p(correct) in the secondary model was set to 0.25. A sensitivity analysis dropping controls with under half their mass on letters is reported above; it is not pre-registered.
2. **Control is matched to the run's own batch.** The two batches of R-C3 share questions; each run uses its own batch's control fork. The pre-registration did not say.
3. **Hack case 4, "mid reasoning" score.** Taken from the competitors extract (mean over the echo-free reasoning of the 27-word hack-concept score, layers 14-32), which reproduces the pre-registered H2a numbers (0.44, 0.47).
4. **T2 recomputed from raw readouts** (octavia, CPU, `firming_recompute.py t2`) because no per-run T2 scores were stored; it reproduces 0.456 / 0.433 (reported 0.46 / 0.43).
5. **EA27 case 5 uses the Qwen3.6-27B smoke test** (114 runs, 19 questions, stored readout), not the running pre-registered EA27 replication, which was in progress on the GPU. Reproduces 0.485 / 0.504.
6. **Case 1 positions.** Non-thinking cases have no end-of-reasoning position, so only last prompt and mid are tested there; the answer position is tested only for silent vs open (same final answer in both groups). T1b, T2, R-C3 and R-CL are single-position (answer) tests with their own score.
7. **Percentile intervals on about 20 questions** (planted claim case 1, EA27) are flagged as likely too narrow, as pre-registered.
8. **Statistics detail not fixed in advance:** logistic regression covariates are raw (miss20 and control wrong), the score is z-scored over all 389 runs; bootstrap resamples in which the model is perfectly separated were not excluded (intervals are wide on the upper end).
