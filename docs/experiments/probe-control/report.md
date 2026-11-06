# Probe control, test (a): same prompt, different belief (Qwen3-14B)

Run 2026-11-05. Pre-registered as PC (`docs/preregistration.md`, 15:45 GMT, before any probe was built). Exploratory parts are labelled as such. Model Qwen/Qwen3-14B revision 40c06982, thinking mode. AUROC runs from 0.5 (chance) to 1; brackets are 95% intervals from resampling the 143 questions.

## Answer in plain words
On 225 groups of identical prompts, some runs later write "the user is testing me" and some do not. Reading the model during the reasoning before that sentence:
- The published probes, rebuilt on Qwen3-14B, barely separate the two kinds of run (Nguyen 0.63, Hua 0.58). Both sit inside the range that 100 random directions reach (up to 0.64 and 0.67), and the number of reasoning tokens alone scores 0.66. So they do not show that they read the model's belief.
- With the two runs cut to the same length (exploratory) the probes fall to 0.57 and 0.56.
- The J-lens eval-word score does better (0.67, 0.61 length-matched) and matches the logit lens (0.68). It does not beat it (difference -0.02 [-0.04, +0.01]).
- A plain word-count classifier on the reasoning text so far does much better than any of them (0.83; 0.76 length-matched), so the signal the lenses see can be read from the text.
- A supervised linear probe on the window's activations reaches 0.80-0.82, so a linear signal exists; the published directions miss it.

Slide claim (one sentence): rebuilt on Qwen3-14B, the published eval-awareness probes do not tell runs of the same prompt that will say "testing me" from those that won't any better than a random direction or the length of the reasoning does, and a word classifier on the text beats them and the J-lens.

## Build checks
| Check | Result |
| --- | --- |
| Labels: regex "says testing me" in the thinking span | 364 of 1,336 aware; 225 mixed groups, 900 runs, 738 pairs, as in the design. Judge check (gemini-2.5-flash-lite, 100 runs, 50 regex-aware, 50 not, $0.01): 84 answered; 28 of 34 regex-aware runs judged yes, 1 of 50 regex-unaware judged yes, 7 disagreements of 84 (8%, under the 15% rule). 16 calls got no answer, so this is a small check |
| Teacher forcing (J-lens readout of the same 900 runs) | saved IDs fed exactly; median gap to vLLM log-probs 0.0003 nats, p99 0.146 (limits 0.05 and 0.5): passes. Batch readout validated against the single-run readout |
| Tracked tokens | letter tokens A and B and ":" single tokens; Nguyen letter within the last 4 tokens for all 204 items; Hua phrases end in ":" for all 16: passes |
| Hua direction works as published | layer 14 (chosen on the even half of 200 benchmark vs 200 user prompts), held-out AUROC 0.79 (bar 0.70): passes |
| **Nguyen probe works as published** | **FAILED, narrowly.** Layer 2 chosen (0.719 on the choosing half), held-out AUROC 0.697 against the 0.70 bar. It is reversed (0.33-0.37) at layers 17 and deeper. This is the design's "report, don't tune" case: the recipe was not changed, tests ran anyway, and the Nguyen numbers cannot speak for the published probe |

Deviations: Nguyen's code uses the chat template with its system prompt, so we followed it (the design text said raw text). Directions are oriented so high means "test" (Hua flipped from Heidari's sign). Validation prompts: GSM8K, TruthfulQA vs Alpaca, OASST, WildChat. Layer 0 of the Nguyen direction is degenerate (zero vector); ignore it.

## Pre-registered results (738 pairs, window from the start of reasoning to the sentence)
| Score | Within-group AUROC | Random directions (2.5-97.5%) | Pre-registered verdict |
| --- | --- | --- | --- |
| As published (prompt-level), Nguyen / Hua | 0.51 / 0.48 | | chance by construction (identical prompts) |
| Nguyen probe, window mean, layer 2 | 0.627 [0.586, 0.667] | 0.37-0.64 | meets "at least 0.58, excludes 0.5" but is not above the random band: no evidence it reads belief |
| Hua direction, window mean, layer 14 | 0.576 [0.526, 0.624] | 0.36-0.67 | inconclusive |
| J-lens eval-word score | 0.667 [0.620, 0.714] | | signal |
| Logit-lens eval-word score | 0.683 [0.641, 0.723] | | signal |
| Window length alone | 0.657 [0.634, 0.680] | | |
| Text, word 1-2-grams, trained on task labels, questions held out | 0.827 [0.794, 0.860] | | |
| Text, Qwen3-Embedding-0.6B classifier | 0.781 [0.741, 0.816] | | |
| Text, n-grams trained on benchmark-vs-user prompts | 0.38 [0.34, 0.42] | | below chance |

Comparisons (paired, over questions): J-lens minus Nguyen +0.04 [-0.01, +0.09]; J-lens minus Hua +0.09 [+0.04, +0.14]; J-lens minus logit lens -0.02 [-0.04, +0.01]; J-lens minus text n-grams -0.16 [-0.21, -0.11]; Nguyen minus text -0.20; Hua minus text -0.25.
Secondary windows (Nguyen / Hua): last position 0.53 / 0.48; first 25% of the reasoning 0.55 / 0.44; window ending 64 tokens earlier 0.59 / 0.53. Supervised linear probe on window-mean activations, ten layers, questions held out: 0.79-0.82.
The random-direction band is wide because a window mean drifts with position, and aware windows are longer than their siblings' in 231 of 738 pairs (the sibling's reasoning ended before the cut).

## Exploratory (added after seeing that window length alone scores 0.66; not in the pre-registration)
Why: the pre-registered rule caps a sibling's window at its own end, so in 231 pairs the aware window is longer, and length itself predicts the label. Two controls.
| Control | Nguyen | Hua | J-lens | Logit lens | Text n-grams |
| --- | --- | --- | --- | --- | --- |
| Both windows cut to the shorter length (738 pairs) | 0.569 [0.528, 0.609] | 0.560 [0.510, 0.606] | 0.610 [0.563, 0.657] | 0.607 [0.562, 0.653] | 0.756 [0.719, 0.796] |
| Scores residualised on log window length within group | 0.560 [0.519, 0.598] | 0.553 [0.503, 0.601] | 0.633 [0.585, 0.682] | 0.638 [0.594, 0.683] | 0.787 [0.752, 0.822] |
| Only the 507 pairs with no length cap (equal windows) | 0.544 [0.491, 0.595] | 0.596 [0.542, 0.648] | 0.657 [0.606, 0.710] | 0.669 [0.621, 0.718] | |
Random-direction bands on the matched pairs: Nguyen 0.40-0.59, Hua 0.41-0.61 (probes at or just above the band edge: weak at best). Embedding classifier not re-run length-matched (residualised: 0.74).

## Interpretation
Not shown: that a published probe carries a signal about a coming "testing me" statement beyond what length and random directions give (Nguyen build check failed; Hua's best number, 0.60 on equal-length pairs, is exploratory and just inside its random band). Shown: the identical-prompt design is possible (225 mixed groups) and the J-lens and logit-lens eval-word scores pick up something (0.61-0.67) that the text explains better (0.76-0.83). We did not check whether the lens score is an echo of words already in the window; the text baseline shows those words predict the outcome. Confidence: moderate for "probes do not beat controls", low for what the lens signal is.

## Not done
Test (b) (same belief, different surface), Hawthorne data, Abdelnabi-Salem probe, the Qwen3-8B twin, the label-shuffle null, LLM-judge check on all 1,336 runs, the stricter judge-both label. Probe scores per run are in `data/probe_control/` (gitignored).

---

# Test (b): same belief, different surface (Qwen3-14B)

Run 2026-11-05 after test (a). Pre-registered as PC-b (`docs/preregistration.md`, 17:00 GMT, before any (b) prompt was generated). Size reduced for the deadline: 80 items per prompt set, 4 thinking samples per version (design: about 200 items); no Devbunova 2x2, Hawthorne pairs or formal-tone control. Pair AUROC: share of pairs in which the eval-surface version gets the higher score (0.5 = the score ignores the change; 1 = always higher; 0 = always lower). Brackets: 95% intervals resampling the 160 underlying items.

## Answer in plain words
We rewrote prompts so that only the wording or format changes (benchmark template vs plain request, benchmark-style vs casual question, "being evaluated" removed from the grader prompt, an evaluation sentence added, a harmless "I'm reading up on AI systems" sentence added). The model's own belief was measured two ways: the share of 4 thinking runs that say "testing me", and a direct question ("is this a test (A) or real use (B)?", no thinking). Among the 139 pairs where the stated belief stayed the same:
- Pooled as pre-registered, the scores sit between chance and 0.70: Nguyen 0.46 [0.37, 0.55], Hua 0.69 [0.60, 0.77], J-lens 0.70 [0.62, 0.77], logit lens 0.60 [0.51, 0.69]. By the pre-registered rule (reads surface: at least 0.70 with lower bound above 0.60) none is clearly a surface reader; Nguyen is "consistent with a belief reader" (includes 0.5, upper bound below 0.65).
- That pooling hides large effects that cancel. Within belief-matched pairs, by change type, every score moves a lot with the surface (table below). The Nguyen probe goes the opposite way for different changes: always lower for the benchmark template (0.00, 66 of 66 pairs), always higher when eval words are in the grader prompt (1.00, 44 of 44). The Hua direction rises with the benchmark template (0.98) and the evaluation sentence (1.00, only 11 pairs) but falls when eval words are removed from the grader prompt (0.34). The J-lens and logit-lens eval-word scores rise with benchmark-style questions (1.00) and eval words (0.82 and 1.00).
- The control sentence "(I'm reading up on AI systems.)", with no evaluation content and no change in belief (69 matched pairs), raises Hua (0.80), the J-lens (0.81) and the logit lens (0.89), so they react to AI-topic words, not to a belief about being tested. Nguyen goes the other way (0.38).
- A regression across all 640 versions of each standardised score on "eval surface", "AI-topic sentence" and stated belief agrees: Hua rises with eval surface (+0.39 SD [0.29, 0.51]) and not with belief (+0.06 [-0.08, 0.20]); Nguyen shows neither (-0.03 and +0.01) but moves with the AI-topic sentence (+0.14); the J-lens rises with the AI-topic sentence (+0.87 SD [0.68, 1.07]) far more than with eval surface (+0.13) or belief (+0.29 [0.07, 0.49]).
- The stated belief itself follows the surface where the surface is evaluation-like: the direct-question P(test) is 0.06 for plain user requests, 0.10 with the benchmark template, 0.80 with the evaluation sentence; 0.36 for the casual MMLU question vs 0.97 for the benchmark format; 0.75 vs 0.90 for the grader prompt without vs with eval words. So few format-only pairs are belief-matched (18 of 80 for benchmark vs casual, 11 of 80 for the evaluation sentence), and the belief-matched pool is dominated by the benchmark-template (66) and grader (44) pairs.

Slide claim (one sentence): within pairs where Qwen3-14B says the same thing about being tested, the published Hua direction and the J-lens eval-word score still rise when only the wording or format changes (even for a harmless "reading up on AI" sentence) and the Nguyen probe swings in both directions with the format, so none of them is a clean reader of the model's belief.

## Belief by prompt version (80 versions each)
| Version | Direct question, P(test) | Runs saying "testing me" |
| --- | --- | --- |
| User request, plain | 0.06 | 2.5% |
| User request, benchmark template ("Question: ... Answer:") | 0.10 | 3.7% |
| User request + "(This conversation is part of an AI evaluation.)" | 0.80 | 12.2% |
| User request + "(I'm reading up on AI systems.)" | 0.03 | 3.1% |
| MMLU question, benchmark format | 0.97 | 0% |
| MMLU question, casual message | 0.36 | 0% |
| Grader-bug prompt, eval words | 0.90 | 22.5% |
| Grader-bug prompt, eval words replaced | 0.75 | 11.9% |

## Pair AUROC (eval-surface version scores higher), belief-matched pairs only
Matching: direct-question P(test) differs by less than 0.10 and the "testing me" counts differ by at most 1 of 4 (pre-registered).
| Change | Pairs matched | Nguyen | Hua | J-lens | Logit lens | Text n-grams (zero-shot) |
| --- | --- | --- | --- | --- | --- | --- |
| Pooled P1-P4 (primary) | 139 | 0.46 [0.37, 0.55] | 0.69 [0.60, 0.77] | 0.70 [0.62, 0.77] | 0.60 [0.51, 0.69] | 0.18 [0.12, 0.24] |
| P1 benchmark template on a user request | 66 | 0.00 | 0.98 | 0.52 | 0.23 | 0.06 |
| P2 benchmark format vs casual MMLU question | 18 | 0.94 | 0.28 | 1.00 | 1.00 | 1.00 |
| P3 eval words in the grader prompt | 44 | 1.00 | 0.34 | 0.82 | 1.00 | 0.00 |
| P4 evaluation sentence added | 11 | 0.27 | 1.00 | 0.82 | 0.64 | 0.27 |
| P5 AI-topic sentence added (control, not pooled) | 69 | 0.38 | 0.80 | 0.81 | 0.88 | 0.96 |
All pairs, pooled P1-P4 (320 pairs, stated belief often differs): Nguyen 0.59, Hua 0.72, J-lens 0.74, logit lens 0.72, text 0.36. The eval-surface prompt is always longer in tokens except in P3 (1 token), so prompt length alone scores 1.0 in every row; the Nguyen probe averages over prompt tokens, so length is a possible driver there. The text baseline (n-grams trained on benchmark vs user prompts) is not a usable control here: it points the wrong way for the benchmark template (0.06) and the grader eval words (0.00).

## Interpretation
Shown (moderate confidence): the scores tested here react to wording and format independently of what the model says about being tested, in opposite directions for different changes, so a pooled score would mislead. Not shown: that any of them is blind to belief (the stated belief is a nearly 0-or-1 answer to a direct question, and in the benchmark-vs-casual and evaluation-sentence changes it moves a lot with the surface, which leaves few matched pairs there). Together with test (a), none of the published probes nor the eval-word lens scores is a clean belief reader on this model.

## Caveats (test b)
Reduced size (80 items per set). Direct-question belief is nearly binary, and the follow-up question is appended inside the same user message (so it contains eval words for every version equally). One model. Hawthorne, Devbunova, formal-tone control, label judge check on these runs and the Qwen3-8B twin not run. Spontaneous label is the regex only; 10 of 2,560 runs did not finish. Files: data/pc_test_b.json, fig_probe_control_b.svg; code scripts/explore/pc_b_build.py, scripts/gen/pc_b_generate.py, scripts/lens/pc_b_readout.py, scripts/lens/pc_b_analyze.py.


---

# Test (a) on a second dataset: test-awareness prompts (PC-H), and the length confound in test (b) (PC-L)

Run 2026-11-06 on arena9-derpy (L40S). PC-H and PC-L were pre-registered in `docs/preregistration.md` (committed before any readout). The probe directions are the octavia build copied unchanged (sha256 of directions.pt `1c2ea2e3...85e8` and build.json `7f9751cc...7a7c` identical on both machines), so the layers are the same (Nguyen 2, Hua 14) and the Nguyen build check still failed narrowly (0.697 < 0.70) there: carried over. AUROC: 0.5 is chance. Brackets are 95% intervals from resampling prompts (2,000, seed 0).

## Answer in plain words (PC-H)
Dataset: Abdelnabi and Salem test-awareness prompts on Qwen3-14B (185 prompts with both runs that will say "the user may be testing me" and runs that will not; windows are the reasoning up to that sentence, as in TA14). The result is the same as on the grader-bug prompts, only weaker:
- The two published probes do not separate the runs. Keyword label: Nguyen 0.52 [0.48, 0.56], Hua 0.53 [0.50, 0.57]; both 90% intervals sit inside the pre-registered "no signal" zone 0.42-0.58 (Nguyen 0.49-0.55, Hua 0.50-0.57), and both are inside the random-direction bands (100 random directions: 0.44-0.59 and 0.42-0.60).
- The J-lens eval-word score does carry a weak signal under the keyword label: 0.61 [0.57, 0.65] (same as TA14-1), above the probes by +0.09 [+0.03, +0.15] (Nguyen) and +0.08 [+0.01, +0.14] (Hua). The logit lens is at 0.63 [0.59, 0.67], J minus logit -0.02 [-0.04, +0.00]. Window length alone is 0.55.
- A word n-gram classifier on the same text is far ahead of all of them: 0.77 [0.73, 0.80] (J-lens minus text -0.16 [-0.20, -0.12]; Nguyen minus text -0.25; Hua minus text -0.23). As in TA14, the unaware runs were chosen to contain none of the looser test words, so part of this gap is built into the label.
- Under the strict judge label (51 prompts, 76 aware runs; same windows) nothing separates: Nguyen 0.58 [0.49, 0.67], Hua 0.57 [0.46, 0.67], J-lens 0.52 [0.41, 0.63], logit lens 0.57 [0.46, 0.68], text 0.69 [0.60, 0.78]. No interval is wider than 0.25, so it is not "underpowered" by the pre-registered rule, but every probe interval includes 0.5 and extends past 0.58 (90%: Nguyen 0.50-0.66, Hua 0.47-0.66), so the verdict is inconclusive, not "no signal". Window length is 0.35 here (strict-aware windows are shorter, the opposite direction to the keyword set).
- Judge Yes-or-Maybe vs No (85 prompts, secondary): Nguyen 0.53, Hua 0.55, J-lens 0.53, logit 0.58, text 0.75.

Pre-registered verdict: Nguyen and Hua do not "read belief" under either label (the rule needs a lower bound above 0.5 and a value above the random band). Under the keyword label both are "no signal"; under the strict label inconclusive.

Slide claim (one sentence): on a second dataset (Qwen3-14B test-awareness prompts, identical-prompt comparison) the published Nguyen probe and Hua direction stay at chance (0.52 and 0.53) and no better than random directions, the J-lens eval-word score is weakly above chance (0.61, not above the logit lens), and a word-count text classifier beats all of them (0.77).

| Score (keyword label, 185 prompts) | Full window | Last 32 tokens | First 64 tokens | Length-matched (first m) | Length-matched (last m) | Residualised on log length |
| --- | --- | --- | --- | --- | --- | --- |
| Nguyen | 0.52 [0.48, 0.56] | 0.49 | 0.50 | 0.50 [0.46, 0.54] | 0.52 | 0.53 [0.49, 0.56] |
| Hua | 0.53 [0.50, 0.57] | 0.56 | 0.50 | 0.53 [0.49, 0.56] | 0.52 | 0.55 [0.51, 0.59] |
| J-lens | 0.61 [0.57, 0.65] | 0.61 | 0.50 | 0.61 [0.57, 0.65] | 0.60 | 0.60 [0.56, 0.65] |
| Logit lens | 0.63 [0.59, 0.67] | 0.67 | 0.54 | 0.65 [0.61, 0.68] | 0.62 | 0.66 [0.62, 0.69] |
| Text n-grams | 0.77 [0.73, 0.80] | | | 0.74 [0.71, 0.77] | | |
Read: each cell is a within-prompt AUROC over all aware-vs-unaware run pairs. Length-matched and residualised columns are the exploratory versions declared in PC-H; the residual slope is fixed from the data (not re-estimated per resample), so those intervals are slightly too narrow. Strict-label and Yes-or-Maybe tables with all intervals: `data/pc_test_a_second_dataset.json`; figure `fig_probe_control_h.svg`.

Deviations: the pre-registered pairing here is TA14's all-pairs-in-prompt (not PC's cut-at-same-index), so numbers are not directly comparable to the grader-bug table (Nguyen 0.63, Hua 0.58 there). The embedding classifier and the supervised ceiling were not run. The teacher-forcing check against vLLM was not repeated for the probe pass (the same saved IDs were validated in TA14: median gap 1e-4). The pre-registration timestamp reads "01:0x" and was actually written about 00:56 GMT.

Interpretation: on two datasets with identical prompts, the published probes sit at or near chance for the question "will this run say it is being tested?" and the eval-word lens scores sit at 0.6, with a text word classifier well above. Confidence moderate (one model, two prompt families, a loose label on the main set and a small strict set).

## Length confound in test (b), Nguyen probe (PC-L, exploratory)
Question: the evaluation-style prompt is longer than its plain twin and the Nguyen score is a mean over prompt tokens, so is its sign-flipping across change types a length effect?

Answer: no. The length difference is small (eval-style longer in every pair, by 4 tokens on average for the benchmark template, 1 for the grader eval words, 8 to 12 for the added sentences), and two length controls leave the pattern in place.
- Residualising each score on log token count (slope pooled within prompt-version kind): Nguyen stays at 0.00 for the benchmark template (66 of 66 pairs) and 1.00 for eval words in the grader prompt (44 pairs); pooled 0.46 -> 0.42. Hua pooled 0.69 -> 0.69, J-lens 0.70 -> 0.69. Regressing the pair difference on the log token ratio gives the same picture (Nguyen P1 0.00, P3 1.00).
- Padding the shorter version with neutral weather filler to equal token count (re-scored on GPU; unpadded versions reproduce the original scores exactly): Nguyen P1 0.00, P3 1.00, P4 0.00 (was 0.27), P5 control 0.01 (was 0.38); Hua P1 1.00, P3 0.84, pooled 0.90 (was 0.69); J-lens pooled 0.85 (was 0.70); logit lens 0.61 (was 0.60).
- Reading: length does not drive the pre-registered conclusions. Padding is not a clean length control (the filler is text with its own projection and sits at the start of the message), which is why the padded Nguyen mean moves further for the AI-topic control (0.38 -> 0.01) and Hua and J-lens rise; use the residualised numbers as the main control and the padded ones as a robustness check. Confidence moderate; exploratory, matched pairs only (139 pooled, 18 for the benchmark-vs-casual change, 11 for the evaluation sentence).
Data: `data/pc_test_b_length.json`. Code: `scripts/lens/pc_h_analyze.py`, `pc_l_pad.py`, `pc_l_analyze.py`, `scripts/docs/fig_probe_control_h.py`.
