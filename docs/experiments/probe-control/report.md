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
