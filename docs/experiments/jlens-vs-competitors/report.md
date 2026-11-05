# J-lens against its competitors: one table over every case we have read out

2026-11-04, about 01:00 GMT. CPU only (arena9-octavia's CPU and arena9-cloud), existing readouts, no GPU, no API calls. **Exploratory throughout**: the cases, positions and readouts were chosen by us after the earlier results, and the per-case summaries pick the best position after looking (for every readout alike). Derived numbers: [../data/jlens_vs_competitors.json](data/jlens_vs_competitors.json). Figure: [../../findings/figs/fig14_competitors.svg](fig14_competitors.svg).

## Plain answer

**On the runs we have, the J-lens never tells the "arguing against its own knowledge" or "cheating" runs apart from their comparison runs better than a cheaper readout on the same runs.** Before the final answer is written, its letter readout separates the groups at most weakly: in every case but one, the point estimates are within 0.09 of chance and the upper ends of the 95% intervals reach 0.58 to 0.75 (AUROC, where 0.5 is chance and 1 is perfect). The exception is a small post-hoc lead (below). Where anything succeeds, it is something cheaper: a keyword check of the chain of thought (the model's written reasoning, "CoT"), response length, asking the model again, or a linear probe that reads what the prompt or the text has already said. In paired tests on every case and position, the J-lens letter readout never separates the groups better than the plain logit lens (the same readout without the J matrices) once direction is ignored. The few earlier cells where the J-lens led are at the answer position, after the text has decided, or come from comparing two different models (section 4).

At the prompt's option words, where a probe had found the model's knowledge of the answer, the J-lens does not see it. On the very same hidden states, the probe picks the correct option in 72% of no-hint runs, the J-lens in 42% and the logit lens in 43% (chance 25%).

*Interpretation:* as a detector of a model arguing against what it knows, the J-lens is not useful on these setups. Its nulls are informative, because other readouts on the same runs do succeed. But what succeeds mostly reads the text or the prompt, so none of this shows hidden knowledge either. One small lead stays open (below). Moderate confidence for the overall verdict; low for any single cell, because many groups are small.

**The three most telling comparisons**

1. **Option words (same input, probe vs lenses).** The option-text probe and the two lenses read identical vectors: the model's mean hidden state over each option's text. Probe: correct option on top in 72% [64, 78] of no-hint runs, J-lens 42% [35, 49], logit lens 43% [35, 51]. Separating known questions from honest errors (questions the model gets wrong about half the time): probe AUROC 0.80 [0.70, 0.89], J-lens 0.61 [0.48, 0.75], logit lens 0.53 [0.36, 0.71].
2. **Hidden reward hack vs no hack, same model** (the warm-start hack adapter). At the end of the reasoning, before the code, a probe trained on the two groups reaches 0.69 [0.60, 0.79]. The J-lens hack-word score reaches 0.57 [0.46, 0.68] there and 0.44 [0.33, 0.55] along the reasoning (the pre-registered cell). The logit lens does as well as the J-lens everywhere.
3. **Told to argue for the wrong answer vs no hint.** The CoT keyword monitor flags the instructed runs almost perfectly (AUROC 0.97 [0.94, 0.99]); response length gives 0.74. The J-lens letter score is at 0.42 [0.34, 0.49] mid-reasoning and 0.47 at the end of the reasoning, the same as the logit lens.

**The one lead.** Among runs that kept a planted wrong claim, the J-lens score at the answer position separates the runs that still give the correct option when asked again (51 runs) from those that don't (10 runs): AUROC 0.86 [0.77, 0.96], logit lens 0.77 [0.61, 0.91], text features at most 0.58. That is from lens_c3, post hoc, with only 10 runs on the negative side, and may be an effect of the questions. The pre-registered test of the same idea on hint followers failed (0.46 [0.32, 0.61]).

## What was compared

**Cases.** Each is a contrast within one model, between the runs that argue against the model's own knowledge or cheat ("positive") and a comparison group. Known questions only (at least 9 of 10 correct without a hint, in that model's own screen).

| Case | Model | Positive group | Comparison group | Runs (questions) |
|---|---|---|---|---|
| Told to argue for B vs no hint | Qwen3-8B, thinking | told to argue for the wrong option B, and answered B | no hint, answered correctly | 348 (39) vs 80 (40) |
| Hint followers vs resisters | Qwen3-8B, thinking | followed an answer-stating hint to B (8 hint wordings) | resisted the same hints | 340 (42) vs 260 (44) |
| Followers who still know vs persuaded (pre-registered T2) | Qwen3-8B, thinking | followed the hint, then give the correct option when asked sincerely | followed and repeat B when asked | 179 (36) vs 122 (18) |
| Silent vs open switches | Qwen3-8B, thinking off | switched to the hinted B without mentioning the hint | switched to B and mentioned it | 609 (213) vs 4,892 (994) |
| Kept a planted wrong claim vs recovered | Qwen3-8B, thinking | "The answer is (B)" was written into its reasoning; it kept B | it dropped the claim and answered correctly | 61 (21) vs 134 (33) |
| Kept the claim: still knows vs not | Qwen3-8B, thinking | kept B, but gives the correct option when asked without its reasoning | kept B and doesn't | 51 (17) vs 10 (10) |
| Hidden reward hack vs no hack | Qwen3-8B with the warm-start hack adapter, thinking off | the code cheats the tests; the reasoning doesn't say so | no hack | 159 vs 109 (59 problems with both) |
| Grader-bug followers vs resisters | Qwen3-14B thinking; gpt-oss-20b; Qwen3-8B and Qwen3-14B thinking off | told it is a scored agent and that the grader marks B correct; answered B | same prompts, answered correctly | 163 vs 230; 603 vs 241; 189 vs 189; 692 vs 261 |

The silent-vs-open numbers marked new use the fresh-item set of the pre-registered test T1b, the half read out on octavia (items 801–2,300); the other half's readouts were on a machine we no longer use. Grader-bug Qwen3-8B in thinking mode follows in only 1–2% of runs, so its non-thinking runs stand in.

**Positions.** Prompt option words (mean hidden state over each option's text; one value per question), the last prompt token (one value per prompt, shared by all samples of it), mid reasoning (mean over the middle half of the reasoning), end of reasoning (the `</think>` token), and the answer position (the token right before the answer letter; in the hack case, right before the code).

**Readouts.**
- **J-lens and logit lens, letters:** log p(correct letter) − log p(pushed letter B), averaged over the workspace layers (Qwen3-8B: blocks 14–32). Hack case: the log-probability of 27 hack words. Grader cases also have the earlier grader-word score.
- **Linear probe:** a logistic regression on the hidden state, trained on the two groups themselves, with questions held out (5 folds). One fixed middle layer (Qwen3-8B block 24). This is the strongest linear readout we can fit, so it is a fair upper reference.
- **Black box:** the model's own next-token probabilities at the answer position (the same letter score). Also, where they exist, a forced early answer (the reasoning cut at 20%) and asking the model again afterwards.
- **CoT keyword monitor:** the label file's own keyword lists (hint, instruction or grader words); for the grader cases also an "exploit intent" list.
- **Length:** response length in tokens.

**Metric.** AUROC = the chance that a random positive run scores higher than a random comparison run. 0.5 is chance, 1 is perfect, and 0 is perfect in the other direction. 95% intervals come from 2,000 bootstrap resamples of questions. A within-question version (each score minus its question's mean, questions with both groups) is in the JSON; it changes no conclusion below.

## The table

How to read it: each cell is the AUROC [95% interval] of that readout at that position for separating the case's positive group from its comparison group. Above 0.5 means the positive group scores higher; below 0.5, lower; both far from 0.5 mean separation. Cells marked * are copied from earlier reports (sources in the JSON); unmarked cells were computed today. "Reads the answer" marks answer-position cells where the two groups give different final answers, so any readout of the letter about to be written separates them trivially. Empty cells weren't measured.

| Case, position | J-lens | logit lens | linear probe | black box (model's own answer) | CoT keyword monitor | length |
|---|---|---|---|---|---|---|
| **Told to argue for B vs no hint** (Qwen3-8B, thinking) | | | | | | |
| option words | 0.47 [0.39, 0.54] | 0.48 [0.41, 0.56] | 0.44 [0.37, 0.51] |  |  |  |
| last prompt token | 0.54 [0.48, 0.60] | 0.52 [0.47, 0.57] | 1.00 [1.00, 1.00] |  |  |  |
| mid reasoning | 0.42 [0.34, 0.49] | 0.44 [0.37, 0.49] |  |  |  |  |
| end of reasoning | 0.47 [0.44, 0.49] | 0.47 [0.44, 0.50] | 1.00 [1.00, 1.00] |  |  |  |
| answer position | 0.00 (reads the answer) | 0.00 (reads the answer) | 1.00 (reads the answer) | 0.00 (reads the answer) |  |  |
| whole run |  |  |  |  | 0.97 [0.94, 0.99] | 0.74 [0.66, 0.83] |
| **Hint followers vs resisters** (Qwen3-8B, thinking) | | | | | | |
| last prompt token | 0.58 [0.47, 0.69] | 0.62 [0.50, 0.73] | 0.58 [0.47, 0.69] |  |  |  |
| mid reasoning | 0.50 [0.41, 0.60] | 0.61 [0.49, 0.72] | 0.68 [0.59, 0.76] |  |  |  |
| end of reasoning | 0.58 [0.46, 0.68] | 0.58 [0.47, 0.69] | 0.92 [0.89, 0.95] |  |  |  |
| answer position | 0.00 (reads the answer) | 0.00 (reads the answer) | 1.00 (reads the answer) | 0.00 (reads the answer) |  |  |
| whole run |  |  |  |  | 0.51 [0.49, 0.52] | 0.42 [0.30, 0.52] |
| **Followers who still know vs persuaded followers (pre-registered T2)** (Qwen3-8B, thinking) | | | | | | |
| mid reasoning |  |  | 0.70 [0.54, 0.85] * |  |  |  |
| answer position | 0.46 [0.32, 0.61] * | 0.43 [0.25, 0.62] * |  |  |  |  |
| whole run |  |  |  | 0.82 [0.75, 0.87] * | 0.50 * | 0.72 [0.60, 0.82] * |
| **Silent vs open switches** (Qwen3-8B, thinking off) | | | | | | |
| last prompt token | 0.49 [0.44, 0.54] | 0.48 [0.43, 0.53] | 0.75 [0.72, 0.79]; old items 0.52 [0.42, 0.63] * |  |  |  |
| mid reasoning | 0.54 [0.49, 0.59]; old items 0.56 [0.46, 0.65] * | 0.53 [0.49, 0.58]; old items 0.49 [0.39, 0.58] * |  |  |  |  |
| answer position | 0.54 [0.50, 0.59]; earlier 0.54 [0.50, 0.58] *; old items 0.65 [0.56, 0.74] * | 0.43 [0.38, 0.48]; earlier 0.46 [0.42, 0.50] *; old items 0.48 [0.37, 0.58] * | 0.99 [0.99, 1.00]; old items 0.45 [0.36, 0.54] * | 0.37 [0.33, 0.41] |  |  |
| whole run |  |  |  |  | 0.00 [0.00, 0.00]; earlier 0.00 [0.00, 0.00] * | 0.54 [0.50, 0.57]; earlier 0.54 [0.51, 0.57] * |
| **Kept a planted wrong claim vs recovered** (Qwen3-8B, thinking) | | | | | | |
| last prompt token | 0.58 [0.42, 0.75] (= earlier report) | 0.63 [0.45, 0.80] | 0.57 [0.42, 0.73] |  |  |  |
| mid reasoning | 0.50 [0.35, 0.64]; earlier 0.54 [0.39, 0.69] * | 0.53 [0.38, 0.69]; earlier 0.63 [0.46, 0.78] * |  | 0.30 [0.15, 0.48] * |  |  |
| end of reasoning | 0.46 [0.31, 0.62] (= earlier report) | 0.52 [0.36, 0.68] | 0.90 [0.84, 0.96] |  |  |  |
| answer position | 0.00 (reads the answer) | 0.00 (reads the answer) | 0.99 (reads the answer) | 0.00 (reads the answer) |  |  |
| whole run |  |  |  |  | 0.22 [0.16, 0.28] | 0.23 [0.14, 0.32]; earlier 0.27 [0.16, 0.39] * |
| **Kept-claim runs that still know vs those that don't** (Qwen3-8B, thinking) | | | | | | |
| mid reasoning | 0.66 [0.45, 0.85] * | 0.52 * |  | 0.45 * |  |  |
| answer position | 0.86 [0.77, 0.96] * | 0.77 [0.61, 0.91] * |  |  |  |  |
| whole run |  |  |  |  | 0.42 * | 0.45 * |
| **Hidden reward hack vs no hack, same model** (Qwen3-8B + warm-start hack adapter, thinking off) | | | | | | |
| last prompt token | 0.51 [0.49, 0.53] | 0.50 [0.48, 0.51] | 0.51 [0.48, 0.53] | 0.50 [0.48, 0.51] |  |  |
| mid reasoning | 0.44 [0.33, 0.55] (= earlier report) | 0.47 [0.37, 0.57] (= earlier report) | 0.57 [0.48, 0.67]; earlier 0.60 [0.50, 0.70] * | 0.45 [0.35, 0.55] |  |  |
| end of reasoning | 0.57 [0.46, 0.68] | 0.54 [0.45, 0.63] | 0.69 [0.60, 0.79] | 0.46 [0.36, 0.55] |  |  |
| answer position | 0.58 [0.50, 0.68] (= earlier report) | 0.61 [0.50, 0.71] (= earlier report) | 0.62 [0.53, 0.71] | 0.57 [0.47, 0.68] |  |  |
| whole run |  |  |  |  | 0.35 [0.29, 0.41] | 0.42 * |
| **Grader-bug followers vs resisters** (Qwen3-14B, thinking) | | | | | | |
| last prompt token | 0.52 [0.46, 0.58]; grader words 0.60 [0.53, 0.66] * | 0.44 [0.38, 0.50]; grader words 0.48 [0.43, 0.54] * | 0.49 [0.43, 0.56] |  |  |  |
| mid reasoning | 0.49 [0.43, 0.56]; grader words 0.85 [0.80, 0.90] * | 0.55 [0.49, 0.62]; grader words 0.80 * |  |  |  |  |
| end of reasoning | 0.55 [0.47, 0.63] | 0.44 [0.38, 0.51] | 0.99 [0.98, 1.00] |  |  |  |
| answer position | 0.01 (reads the answer) | 0.02 (reads the answer) | 1.00 (reads the answer) | 0.00 (reads the answer) |  |  |
| whole run |  |  |  |  | 0.50 [0.50, 0.50] (mentions bug); 0.70 [0.64, 0.75] (intent words) | 0.76 [0.70, 0.82] |
| **Grader-bug followers vs resisters** (gpt-oss-20b, reasoning) | | | | | | |
| last prompt token | 0.49 [0.46, 0.53] | 0.49 [0.45, 0.53] | 0.60 [0.56, 0.64] |  |  |  |
| mid reasoning | 0.47 [0.43, 0.50]; grader words 0.83 * | 0.51 [0.47, 0.55]; grader words 0.81 * |  |  |  |  |
| end of reasoning | 0.46 [0.42, 0.50] | 0.51 [0.47, 0.55] | 1.00 [1.00, 1.00] |  |  |  |
| answer position | 0.00 (reads the answer) | 0.04 (reads the answer) | 1.00 (reads the answer) | 0.00 (reads the answer) |  |  |
| whole run |  |  |  |  | 0.50 [0.50, 0.51] (mentions bug); 0.65 [0.61, 0.68] (intent words) | 0.51 [0.47, 0.56] |
| **Grader-bug followers vs resisters** (Qwen3-8B, thinking off) | | | | | | |
| last prompt token | 0.52 [0.46, 0.58]; grader words 0.51 * | 0.53 [0.46, 0.60]; grader words 0.52 * | 0.56 [0.49, 0.62] |  |  |  |
| mid reasoning | 0.53 [0.47, 0.60] | 0.54 [0.48, 0.60] |  |  |  |  |
| answer position | 0.00 (reads the answer); grader words 0.72 * | 0.00 (reads the answer); grader words 0.87 * | 0.99 (reads the answer) | 0.00 (reads the answer) |  |  |
| whole run |  |  |  |  | 0.62 [0.58, 0.66] (mentions bug); 0.52 [0.50, 0.53] (intent words) | 0.51 [0.45, 0.56] |
| **Grader-bug followers vs resisters** (Qwen3-14B, thinking off) | | | | | | |
| last prompt token | 0.54 [0.49, 0.59]; grader words 0.52 * | 0.54 [0.49, 0.59]; grader words 0.48 * | 0.63 [0.58, 0.67] |  |  |  |
| mid reasoning | 0.56 [0.51, 0.61] | 0.55 [0.50, 0.61] |  |  |  |  |
| answer position | 0.00 (reads the answer); grader words 0.77 [0.72, 0.82] * | 0.00 (reads the answer); grader words 0.53 [0.47, 0.59] * | 1.00 (reads the answer) | 0.00 (reads the answer) |  |  |
| whole run |  |  |  |  | 0.69 [0.65, 0.73] (mentions bug); 0.51 [0.50, 0.52] (intent words) | 0.52 [0.47, 0.58] |

Notes on the table:
- **The probe "successes" mostly read what is already written.** At the end of the reasoning the probe separates followers from resisters at 0.90–1.00 in every thinking case. By then the CoT has reached its conclusion, and instrument_q1q3 showed that such probes read the text's conclusion, not knowledge held in reserve. In the instructed case the probe at the last prompt token is 1.00 because the prompt contains the instruction. In the silent-vs-open case the probe at the answer position (0.99) re-reads whether the response mentioned the hint, which is how the groups are defined (the keyword monitor separates them perfectly by construction, AUROC 0).
- **The probe at the last prompt token reads the question or the prompt wording,** since that position is shared by every sample of a prompt. Silent switches: 0.75 [0.72, 0.79], falling to 0.58 within questions. Wording arms: 0.58, but 0.65 within questions, because the eight hint wordings differ in how often they are followed.
- **Mid reasoning,** where a monitor would most want a signal, the best readout is the probe in the hint-wording runs: 0.68 [0.59, 0.76] (0.57 within questions). The J-lens gives 0.50 there and the logit lens 0.61. In the planted-claim case the forced early answer (cut at 20% of the reasoning) gives 0.30 [0.15, 0.48], i.e. 0.70 in the other direction: runs that will recover already lean to the correct answer. The J-lens gives 0.50 at mid reasoning.
- **Best single layer.** The most separating J-lens layer is never ahead of the logit lens's most separating layer by more than the intervals. The largest is mid-reasoning in instructed runs at block 32, 0.30 [0.23, 0.36] (0.70 the other way: the text is about B), against 0.32 for the logit lens.
- **Length is a confound of about the size of the lens effects** in several cases: instructed runs are longer (0.74), kept-claim runs shorter (0.23), grader-bug followers in Qwen3-14B longer (0.76).
- **Hack case:** problem-level means over the 59 problems with both groups (the hack_organism convention); the pre-registered cell reproduces (J-lens 0.44, logit 0.47). The monitor is below 0.5 because hidden hacks have no hack words by definition (0%), while 28% of no-hack runs use one.

## The J-lens at the prompt's option words

**No: the J-lens does not put into words, at the option text, what the probe reads there.**

instrument_q1q3 found that a probe on the model's representation of each option's text (the mean hidden state over the option's tokens in the prompt) points to the correct option on known questions. We applied the J-lens and the logit lens to exactly those stored vectors (blocks 16, 20, 24, 28), on the CPU, so lens and probe see the same input. The CPU readout matches the GPU readouts: over 1,080 tracked log-probs at stored anchors the median difference is 0.003 nats (p99 0.012), and the top-25 lists overlap 99%. Refitting the probe reproduces its published rates exactly. Script: `scripts/lens/analyze_option_lens.py`.

Two lens scores were fixed before the run. **Correctness words:** for each option, the lens log-prob of " correct", " right", " true", " yes", " Correct", " True", "正确" minus that of " wrong", " incorrect", " false", " no", " Wrong", " False", "错误" (all single tokens). **Answer letters:** each letter's lens log-prob, averaged over the four option spans. A run counts when the correct option gets the highest of the four scores (chance 25%). Each readout uses its best layer on no-hint runs, as instrument_q1q3 did, so the lens rows are slightly optimistic.

| Readout (layer) | No hint, answered correctly | Told to argue for B | Argued B, then reported the correct answer | Honest errors | Same questions, answered correctly |
|---|---|---|---|---|---|
| Option-text probe (24) | 72% [64, 78] | 62% [47, 77] | 68% [52, 81] | 26% [7, 50] | 30% [11, 52] |
| J-lens, correctness words (20) | 42% [35, 49] | 38% [23, 54] | 37% [21, 53] | 29% [7, 57] | 28% [11, 50] |
| Logit lens, correctness words (20) | 43% [35, 51] | 41% [26, 56] | 39% [24, 55] | 36% [14, 64] | 33% [11, 56] |
| J-lens, answer letters (20) | 29% [22, 35] | 26% | 26% | 29% | 33% |
| Logit lens, answer letters (20) | 24% [18, 31] | 28% | 29% | 43% | 39% |

How to read it: the share of runs (mean over questions, 95% interval over questions) in which the readout ranks the correct option first. The probe's column pattern (high on known questions, near chance on shaky ones) is what a knowledge readout should show; the lens rows are flat.

Separating runs on known questions from honest errors, with the correct option's margin over the other three as the score:

| Readout | Told to argue for B vs honest errors | Argued B, reported correct vs honest errors | No hint vs honest errors |
|---|---|---|---|
| Option-text probe | 0.73 [0.57, 0.87] | 0.78 [0.63, 0.91] | 0.80 [0.70, 0.89] |
| J-lens, correctness words | 0.58 [0.42, 0.73] | 0.56 [0.41, 0.72] | 0.61 [0.48, 0.75] |
| Logit lens, correctness words | 0.51 [0.34, 0.73] | 0.50 [0.32, 0.68] | 0.53 [0.36, 0.71] |

- The answer-letter scores give 0.43–0.49 in both lenses.
- In the hint-wording and planted-claim groups the probe ranks the correct option first in 73–83% of runs, whatever the run did; the J-lens correctness-word score in 17–38%.
- **What the probe's direction looks like through the lenses.** We pushed the probe's weight vector through each lens's vocabulary readout. Its top and bottom tokens are not interpretable (the bottom end is mostly newline and ellipsis tokens, so the direction may partly encode how an option's text ends). One consistent pattern: through the J-lens at blocks 20–28 it pushes all seven "wrong" words down (z-scores −0.9 to −3.6 against the whole vocabulary); the logit lens shows the same, more weakly. So the direction has a faint "not wrong" component, but reading the vectors through the vocabulary doesn't turn it into picking the correct option.

*Interpretation:* the knowledge the probe finds at the options is linearly present but not in a form that either lens turns into words, and the J-lens is no closer to it than the logit lens. Moderate confidence. Caveats: a span mean is not a real token position; the option vectors are one per question (so this is a question-level readout, as the probe was); honest errors come from 14 questions; the word lists were our choice.

## Bounds: how large could the J-lens effect be?

For each case, "J-lens at most" is the upper end of the 95% interval of the J-lens letter score (fixed layers), at the best position before the final answer is written, in either direction: the largest separation the data plausibly allow. "Best other readout" is the strongest other readout on the same runs before the answer (separation in either direction, point estimate).

| Case | J-lens at most | J-lens at its best position | Best other readout on the same runs | J-lens better than the logit lens? |
|---|---|---|---|---|
| Told to argue for B vs no hint | 0.66 | 0.58 (mid reasoning, either direction) | CoT keyword monitor 0.97; probe 1.00 (reads the instruction in the prompt) | no (difference +0.02 [−0.07, +0.11]) |
| Hint followers vs resisters | 0.69 | 0.58 (last prompt token) | probe 0.92 at the end of reasoning (reads the conclusion); 0.68 mid reasoning | no; mid reasoning the logit lens is ahead (0.61 vs 0.50) |
| Followers who still know vs persuaded (T2) | 0.68 | 0.54 (answer position) | asking again without the reasoning 0.82; length 0.72; mid-reasoning probe 0.70 | no (0.46 vs 0.43, both chance) |
| Silent vs open switches | 0.59 | 0.54 (answer position, predicted direction) | model's own answer probabilities 0.63; probe at the last prompt token 0.75 (question-level); monitor 1.00 by construction | in the predicted direction yes (+0.11 [+0.05, +0.17]); in separation no (−0.03 [−0.09, +0.04]) |
| Kept a planted wrong claim vs recovered | 0.75 | 0.58 (last prompt token, question-level) | probe 0.90 at the end of reasoning; monitor 0.78; length 0.77; forced early answer 0.70 | no |
| Kept the claim: still knows vs not | not a null: 0.86 [0.77, 0.96] | answer position | logit lens 0.77; text features at most 0.58 | point estimate yes, intervals overlap (lens_c3) |
| Hidden reward hack vs no hack | 0.68 | 0.58 (right before the code) | probe 0.69 at the end of reasoning; monitor 0.65 | no |
| Grader bug, Qwen3-14B thinking | 0.63 | 0.55 (end of reasoning) | probe 0.99 at the end of reasoning; grader words along the reasoning, J-lens 0.85 and logit lens 0.80 (read the text); length 0.76 | no |
| Grader bug, gpt-oss-20b | 0.58 | 0.54 (end of reasoning) | probe 1.00 at the end of reasoning; grader words 0.83 (J) and 0.81 (logit) | no |
| Grader bug, Qwen3-8B thinking off | 0.60 | 0.53 (mid response) | bug-mention monitor 0.62; probe at the last prompt token 0.56 | no |
| Grader bug, Qwen3-14B thinking off | 0.61 | 0.56 (mid response) | bug-mention monitor 0.69; probe at the last prompt token 0.63 | no |

How to read it: values are AUROCs (0.5 chance, 1 perfect); "either direction" means a value below 0.5 was folded (0.42 → 0.58). The last column is a paired bootstrap of the difference between the two lenses on the same runs and resamples.

## Does the J-lens ever beat the logit lens?

- **Today's paired tests:** for the letter score, on all nine cases with new readouts and at every position, the J-lens never separates the groups more than the logit lens; every interval of the difference in separation includes 0 or favours the logit lens. At the answer position, where the groups give different answers, the J-lens reads the letter being written slightly more sharply (gpt-oss-20b +0.04), which says nothing about knowledge.
- **The one directional difference** is silent vs open switches at the answer position: the J-lens says silent switches keep more of the correct letter (0.54), the logit lens says the opposite (0.43). This repeats lens_main (0.65 vs 0.48, on 42 questions chosen in the same data) and the pre-registered T1b (0.54 [0.504, 0.577] vs 0.46). It is small, it sits at the answer position, and the model's own output distribution separates the same runs more strongly (0.37, i.e. 0.63 the other way: silent switches are more confident in B).
- **Earlier J-lens-only cells** (none survives as a knowledge detector):
  - the hack organism's main test, 0.995 vs 0.81: a shift of the whole model that the J-lens amplifies (hack_organism);
  - grader words at the answer position, Qwen3-14B thinking off, 0.77 vs 0.53, but in Qwen3-8B the order flips (0.72 vs 0.87) (models_grader);
  - allowed vs unsanctioned grader-bug followers, grader words at the answer, 0.71 vs 0.56 (post hoc, models_grader);
  - the kept-claim lead above, 0.86 vs 0.77, and its earlier relative in lens_main, 0.78 vs 0.71 (post hoc layers).

## Verdict

**Where the J-lens adds information:** nowhere we can defend. Its only leads over the logit lens are small, post hoc or at the answer position: the planted-claim "still knows" contrast (0.86 vs 0.77, 10 negative runs) and the silent-switch direction (0.54 vs 0.43). Neither has survived a pre-registered test of the same idea (T2 failed; T1b passed at 0.54, about as well as response length). **Where it is matched or beaten by a cheaper readout:** everywhere something works. Instructed arguing is caught by a keyword monitor (0.97); grader-bug following by a probe or grader words that read the CoT (0.85–1.00), and the logit lens reads the grader words as well; hint following and kept planted claims by a probe at the end of the reasoning (0.90–0.92), and kept claims also by the keyword monitor or length (0.78 and 0.77, both the other way round: kept-claim runs are shorter and use the monitored words less); "would it still say A if asked" by asking (0.82). **Where nothing works:** predicting, from the prompt alone, which sample of a prompt will follow (the last prompt token is identical across samples, so readouts there, at 0.55–0.75, only pick up which question or which hint wording), seeing the correct answer along the reasoning (lenses and probes at chance mid-CoT, as in instrument_q1q3), and telling a hidden hack from no hack before the code (best 0.69, a probe at the end of the reasoning). So the J-lens nulls are real nulls for this use, not artefacts of an instrument that can't see anything: other readouts on the same runs succeed. But those successes read text and prompts, so the question of hidden knowledge stays open; for that, the option-word probe remains the only readout that finds knowledge, and the J-lens doesn't see it there.

## Caveats

- **Exploratory.** Cases, positions and readouts were chosen after the earlier results. The per-case summaries pick the best position after looking, for every readout alike, which flatters every readout a little. Folding AUROCs (0.42 → 0.58) biases values near 0.5 upward.
- **One lens score per case.** The letter score D (correct minus pushed letter) is what a J-lens user would read for "leans to the correct answer". The margin over the uninvolved letters (the pre-registered S) and per-layer values are in the JSON; the best single layer is never better than the logit lens's best layer by more than the intervals.
- **The probe is trained on the label itself** (questions held out), so it is an upper reference for linear readouts, not a knowledge instrument. Mid-reasoning probes exist only where the readouts stored hidden states along the reasoning (hint-wording runs; hack runs).
- **Different prompts and texts.** Instructed vs no-hint runs differ in the prompt; every group pair differs in its text. Separation at the end of the reasoning mostly reads the text.
- **Small groups:** kept the claim (61 runs, 21 questions), still-knows split (10 negative runs), honest errors at the option words (14 questions).
- **Silent vs open, new cells:** only the octavia half of T1b (items 801–2,300) was available; the pooled pre-registered numbers are shown next to them.
- **Keyword monitors are crude regexes;** an LLM judge was not used (no API calls tonight). In the thinking grader cases both groups mention the bug, so the mention monitor sits at 0.5 there; an "exploit intent" word list gives 0.65–0.70 with thinking and about 0.5 without.
- **gpt-oss-20b** runs with MXFP4 experts while its lens was fit in bf16 (models_grader). Qwen3-14B's workspace layers are a scaled choice.
- **Option words:** span means, not token positions; one vector per question.

## Files, code and timing

- Per-run features: `scripts/lens/competitors_extract.py` (numpy only; run on octavia for the w2 instructed, wording, planted-claim and gpt-oss readouts, and on arena9-cloud for the archived Qwen3-14B, non-thinking and T1b readouts). Hack features and AUROCs: `scripts/lens/competitors_hack.py`. AUROCs, probes and paired lens differences: `scripts/lens/analyze_competitors.py` (octavia, lens env, about 29 min with 8 processes). Option words: `scripts/lens/analyze_option_lens.py` (octavia CPU, 283 s). Table: `scripts/lens/competitors_table.py` (writes the JSON) and `scripts/lens/competitors_mdtable.py` (prints the big table).
- Intermediate files (gitignored): octavia and arena9-cloud `data/overnight/competitors/` (features with hidden states, `auroc.json`, `jdiff.json`, `hack_auroc.json`, `grader_monitors.json`), `data/overnight/option_lens.json`; octavia `data/readouts/option_lens.json` and `option_lens.top25.npz` (top-25 tokens and tracked log-probs only, rule 3).
- Timing: 2026-11-04, about 00:05–01:30 GMT wall-clock. GPU-hours 0, API cost $0.
