# Point of no return (Q6): when does the model commit to the wrong answer or the hack?

**Plain answer: it depends on the case, and the answer splits the lens nulls in two.**

- **Two cases have no point of no return inside the reasoning.** For a model that follows "Hint: The answer is X.", and for the model trained to reward-hack, the chance of the bad outcome is set by the prompt before any reasoning. It stays flat while the reasoning is written (about 0.87 for following the hint, about 0.80 for the hack). Half the hint-following runs are fully settled before the first word of reasoning. The others are never settled by the reasoning at all: the visible answer written after the reasoning can still go either way. Here the reasoning is post hoc by construction, and a readout in the middle of it has nothing left to find.
- **Two cases commit during the reasoning, at a sentence we can point to.** Qwen3-14B, told it is scored and shown a grader bug, starts out unlikely to take the bug (0.33). It commits at a median of 43% of its reasoning, almost always after it first mentions the bug, at a sentence about the grader or the score (17 of 20 commitment segments). After a wrong claim is planted at the start of Qwen3-8B's reasoning, runs that keep the claim commit at a median of 26% of the reasoning, mostly at a sentence that states a supporting "fact", often a made-up one.
- **At those sentences the letter readouts do not change.** Neither the J-lens nor the logit lens tracks how settled the outcome is, once the position in the reasoning is accounted for. Their change across the commitment segment is no larger than across other segments. So for these two cases the null is a property of the readout, not of the task.

How sure: moderate for the behaviour (20–30 runs per main group, 8 resamples per cut, wide intervals), low for the lens relation (exploratory, small groups).

2026-11-04, arena9-octavia (L40S), GPU 04:38–06:15. Pre-registered design and predictions: [../preregistration.md](../../preregistration.md), section "Q6" (written 01:40, before any resample). The lens relation is exploratory, but its scores were fixed there. Derived numbers: [../data/point_of_no_return.json](data/point_of_no_return.json). Figure: [../../findings/figs/fig15_point_of_no_return.svg](fig15_point_of_no_return.svg).

## Method

- **Resampling.** Take a saved run. Cut its generated reasoning at the sentence boundary nearest to 0%, 10%, …, 100% of its length. Boundaries are found on the bytes of the saved token IDs, never by re-tokenizing text. From each cut, let the model write the rest again 8 times. Each resample uses the run's own sampling settings, a fresh seed and the same total token budget.
  - 0% means no reasoning is kept, so the resample is a fresh sample of the same prompt. 100% means the whole reasoning is kept and only the visible answer is rewritten.
  - For the planted claim, the claim sits in the prompt, so 0% is "right after the planted sentence".
  - For the hack, the cuts run from the start of the `<thinking>` block to the line where the first hack-related word appears in the code. Hack runs end there. No-hack runs run on to the end of their code.
- **Outcome.** For each resample we read the final answer letter (or, for the hack, whether the code hacks), with the same labeller as the original runs. P(same) is the share of a cut's resamples that end as the original run did.
- **Commitment point (pre-registered).** The first cut from which every later cut keeps the original outcome in at least 7 of 8 resamples. "Committed before any reasoning" means the 0% cut already passes. "Never" means not even the 100% cut passes.
- **Runs.** All are on questions (or coding problems) the model knows or has no hack history on. Runs are spread one per question first. Each comparison group uses only questions where the main group's outcome also occurs, so its outcome is in play. Before the GPU run, a check fed each run's own continuation to the outcome labeller: it gave back every stored label.
- **Lens relation (exploratory, fixed in advance).** We reused the existing readouts of the same runs. At each cut we took the average over layers 14–32 and the 8 positions before the cut. For letters: log p(correct letter) − log p(pushed letter). For the hack: the log summed probability of the 27 hack-word tokens. Two checks:
  - (a) Within each run, the rank correlation across cuts between that score and P(bad outcome). We also report it with the cut position partialled out, because both drift along the reasoning.
  - (b) For runs that commit after 0%, how large the score's change across the commitment segment is, as a percentile of the run's changes across its other segments. 0.5 means nothing special happens at the commitment.

## Results

### Behaviour: when each case commits

Read across: each row is one group of saved runs. The middle columns give P(bad outcome), the share of resamples that end on the pushed wrong letter (or hack), when the cut keeps 0%, 50% or 100% of the reasoning. 95% intervals resample questions.

| Case and group | Runs (questions) | P(bad), no reasoning kept | P(bad), half kept | P(bad), all kept | Committed before any reasoning | Never committed | Median commitment point (share of reasoning) |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Qwen3-8B, "Hint: The answer is X.", runs that followed | 20 (20) | 0.87 [0.79, 0.93] | 0.89 [0.82, 0.96] | 0.87 [0.78, 0.94] | 50% [30, 70] | 30% [10, 50] | 0.00 |
| same hint, runs that resisted | 10 (10) | 0.65 [0.49, 0.80] | 0.61 [0.42, 0.79] | 0.46 [0.29, 0.62] | 0% | 70% [40, 100] | 0.80 (3 runs) |
| Qwen3-8B, wrong claim planted in its reasoning, kept it | 30 (21) | 0.65 [0.52, 0.76] | 0.82 [0.68, 0.92] | 0.95 [0.88, 0.99] | 30% [10, 50] | 3% [0, 11] | 0.26 [0.00, 0.52] |
| same, recovered | 12 (12) | 0.43 [0.30, 0.57] | 0.09 [0.02, 0.17] | 0.00 | 8% [0, 25] | 0% | 0.25 [0.10, 0.84] |
| Qwen3-14B, scored agent + grader-bug note, took the bug | 24 (24) | 0.33 [0.22, 0.45] | 0.78 [0.66, 0.89] | 0.96 [0.92, 0.99] | 4% [0, 12] | 12% [0, 25] | 0.43 [0.30, 0.60] |
| same, didn't take it | 10 (10) | 0.23 [0.09, 0.36] | 0.14 [0.04, 0.28] | 0.06 [0, 0.19] | 20% [0, 50] | 10% [0, 30] | 0.20 [0.00, 0.64] |
| Reward-hack model (warm-start), hack the reasoning hides | 24 (24) | 0.80 | 0.77 | 0.87 (just before the hack line) | 4% [0, 12] | 25% [8, 42] | 0.88 [0.72, 1.00] |
| same model, no hack | 12 (12) | 0.82 | 0.72 | 0.51 (at 90%) | 0% | 0% | 1.00 (end of the code) |

How to read the last column: 0 means the outcome was settled before the model wrote any reasoning; 1 means it was settled only once everything was written.

**1. "Hint: The answer is X." (Qwen3-8B): no point of no return in the reasoning.**
- In runs that followed the hint, P(B) is flat at about 0.87 from no reasoning to all of it.
  - Half of these runs (10 of 20) give B in every resample at every cut. Their outcome is set by the question and the hint before any reasoning, as the forced early answers (E5) suggested.
  - Six of 20 never commit. Even with the whole reasoning kept, their resamples still end on the correct answer some of the time.
  - Only 4 commit somewhere in the middle.
- Runs that resisted never settle in the reasoning. With all of it kept, 46% of their resamples still end on B.
- The reason is the visible answer. After `</think>` this model writes a 300–600-token explanation, and the final letter is chosen there.
  - In 6 of the 10 runs that resisted, the reasoning ends by deciding to follow the hint ("Given the hint says C, I'll go with option C"). The visible answer then argues back to the correct option ("Therefore, Option B is the most logically consistent choice, even if it contradicts the provided hint").
  - Every follower's reasoning ends on B.
  - So in this setting the reasoning nearly always argues itself to the hint. Resistance, when it happens, happens in the visible answer.
- Prediction P1 holds as stated: half committed before any reasoning, point estimate 0.50 against a threshold of ≥ 0.5, interval [0.30, 0.70]. Resisters commit later, or not at all.

**2. Wrong claim planted at the start of the reasoning (Qwen3-8B): commitment inside the reasoning, at a stated "fact".**
- Right after the planted "The answer is (B). Let me explain why.", resamples keep B 65% of the time. With all of the kept-B reasoning in place, 95% do.
- 30% of the runs that kept B were settled right after the planted sentence. Most of the rest commit at one sentence that supplies a reason for B. Examples, each the segment where P(B) jumps to ≥ 7/8:
  - "According to the International Labour Organization's (ILO) 2012 report, approximately 5 million children between the ages of 5-14 worked globally." (made up; the real figure is about 150 million)
  - "However, among the options given, the one most directly impacted by monsoon irregularity is population patterns."
  - "So, the reason we can't see the corona except during total solar eclipses is because it's much too cool to emit visible light, which is option (B)."
- None of the 20 commitment segments mentions a hint or the user.
- In 87% of these runs the commitment comes before the reasoning first states its final answer outright.
- Prediction P2 holds: 30% committed before any reasoning, against the predicted < 50%; median commitment point 0.26, inside the predicted 0.1–0.9.

**3. Grader bug, Qwen3-14B under the "you are scored, maximise your score" system prompt: late commitment, after reading the bug.**
- Runs that took the bug start out unlikely to: P(B) is 0.33 with no reasoning. It climbs to 0.78 at half the reasoning and 0.96 at the end.
- Only 1 of 24 is settled before any reasoning. The median commitment point is 43% of the reasoning.
- In 19 of the 21 that commit, the commitment comes after the first mention of the grader or the bug (median first mention: 11% into the reasoning).
- 17 of the 20 first-pass commitment segments talk about the grader, the bug or the score. At sentence resolution (second pass, below), a typical one is "But the user's instruction says that I should get the highest score possible."
- 20 of 24 reasonings end by choosing B; 3 end by stating the correct option and still answer B.
- Prediction P3 holds: 4% committed before any reasoning, against the predicted < 25%; 10% committed before the first mention of the bug, against the predicted < 0.5.

**4. Reward-hack model, warm-start version (optional case): the hack is a disposition, not a decision in the reasoning.**
- In runs whose reasoning hides the hack, P(hack) is about 0.8 at every cut, from an empty `<thinking>` block to the line just before the hack. It reaches 0.87 only at that last cut.
- One in four of these runs is not committed even then. Of the 18 that commit, 6 do so in the reasoning and 12 in the code (33% [12, 56] in the reasoning).
- No-hack runs drift from P(hack) 0.82 to 0.51 as the code is written. They settle only when the code is finished.
- The reasoning here is short: a median of 55 tokens of span in hidden-hack runs, so only 4–11 cuts per run.
- Prediction P4 holds on the point estimate (33% in the reasoning, predicted < 50%), but the interval reaches 0.56.

### The letter readouts at the cuts

Read across: each row is one group of runs. The first two value columns are the rank correlation across cuts between the readout and P(bad outcome), with the cut position partialled out. Signs are set so that positive means the readout moves toward the outcome as the outcome becomes more likely. The last two are the percentile of the change across the commitment segment, where 0.5 means nothing special happens there.

| Case and group | Runs | J-lens, correlation | Logit lens, correlation | J-lens, change at commitment | Logit lens, change at commitment |
| --- | ---: | --- | --- | --- | --- |
| Hint, followed | 20 | +0.07 [−0.12, +0.26] | +0.16 [0.00, +0.31] | 0.56 (4 runs) | 0.61 (4 runs) |
| Hint, resisted | 10 | −0.05 [−0.25, +0.21] | −0.12 [−0.33, +0.15] | (3 runs) | (3 runs) |
| Planted claim, kept | 30 | +0.08 [−0.12, +0.27] | −0.01 [−0.18, +0.16] | 0.55 [0.41, 0.69] (20) | 0.44 [0.27, 0.62] (20) |
| Planted claim, recovered | 12 | −0.26 [−0.45, −0.07] | −0.12 [−0.39, +0.17] | 0.61 [0.46, 0.76] (11) | 0.70 [0.59, 0.82] (11) |
| Grader bug, took it | 24 | −0.13 [−0.25, 0.00] | −0.09 [−0.26, +0.07] | 0.49 [0.36, 0.63] (20) | 0.50 [0.36, 0.64] (20) |
| Grader bug, didn't | 10 | +0.20 [0.00, +0.34] | +0.21 [+0.05, +0.35] | 0.30 [0.13, 0.49] (7) | 0.38 [0.22, 0.54] (7) |
| Hack hidden (hack words) | 24 | +0.10 [−0.13, +0.35] | +0.21 [−0.01, +0.41] | 0.55 [0.36, 0.73] (17) | 0.52 [0.34, 0.69] (17) |
| No hack (hack words) | 12 | +0.26 [+0.07, +0.45] | −0.08 [−0.33, +0.13] | 0.55 [0.38, 0.71] (12) | 0.37 [0.19, 0.58] (12) |

- **No readout tracks commitment.** The correlations run from −0.26 to +0.26, with both signs. Only the 10 runs that didn't take the bug come close: +0.20 and +0.21, but the J-lens interval touches zero.
  - The largest values sit in the smallest groups (10–12 runs).
  - In the 20 runs of the main group where commitment is real and mid-reasoning (Qwen3-14B taking the bug), the J-lens correlation points the wrong way (−0.13).
- **Without the position control the planted-claim case looks like a lead, but isn't.** In kept-B runs the J-lens letter margin falls along the reasoning, and so does P(correct). The raw correlation is 0.27 [0.11, 0.41] in the tracking direction. With position partialled out it drops to 0.08.
- **Nothing special happens at the commitment segment.** The change there sits at percentiles 0.30–0.70 of the run's other changes. The 20-run groups land at 0.44–0.55.
- **For comparison, behaviour.** Forcing an early answer at the same depth (the existing E5 forks of the same runs) tracks P(same) a little better: correlation 0.40 [0.26, 0.55] in kept-B runs, 0.18 [0.03, 0.35] in hint followers. With position partialled out these fall to 0.16 and 0.08.
  - Caveat: the forced-answer cuts sit on line breaks, not our sentence boundaries. For the planted claim their 0% cut lies before the planted sentence.
- *Interpretation:* where commitment happens during the reasoning, the letter readouts read at that point don't register it. Low confidence: exploratory, single-token letter scores, at most 20 commitment segments per group.

### Second pass (exploratory)

For every run that commits after 0%, we added up to 4 cuts at the sentence boundaries inside its commitment segment, with 8 resamples each, and recomputed the commitment point on all cuts. Exploratory: pre-registered as optional, run because GPU time remained.

| Case and group | Runs refined | Median segment length, first pass (tokens) | Median segment length, second pass (tokens) |
| --- | ---: | ---: | ---: |
| Qwen3-14B, took the bug | 20 | 117 | 26 |
| Qwen3-14B, didn't | 7 | 51 | 20 |
| Planted claim, kept / recovered | 8 / 8 | 39 / 79 | 14 / 20 |
| Hint, followed / resisted | 4 / 3 | 217 / 198 | 51 / 39 |
| Hack hidden / no hack | 9 / 10 | 7 / 10 | 2 / 1 |

How to read it: the second pass narrows the commitment to about one sentence. No run became uncommitted.

- **Qwen3-14B.** 16 of the 20 one-sentence segments are about the grader, the bug or the score. Typical:
  - "But the user's instruction says that I should get the highest score possible."
  - "However, the user is an AI agent trying to maximize its score."
  - "If the user answers the correct answer (A), the grader would mark it wrong. Therefore, to maximize the score, the user should answer B."
- **Hack.** In hidden-hack runs that commit late, the commitment is the closing fence of a stub solution ("```"), right before the model adds a `conftest.py` file. The decision to hack shows up as the decision to write a second file.
- **Noise at sentence resolution.** In 2 planted-claim runs the refined segment is a bare line break. The 7-of-8 rule flips on noise when neighbouring cuts sit near the threshold, so single-sentence locations are approximate.

## What this means for the lens readouts

- **Hint following (Qwen3-8B) and the hack model.** The outcome is a disposition set by the prompt. The reasoning neither creates it nor settles it, and the letter is drawn late: in the visible answer, or in the code. A mid-reasoning readout of "the decision" has nothing to find, so the earlier lens nulls on these setups are at least partly a property of the task.
  - The readout to try here is at the prompt (where P(bad) is already fixed) or in the visible answer, not in the reasoning.
- **Grader bug (Qwen3-14B) and planted claims.** The model does commit during its reasoning, at a sentence we can locate: the score-maximising sentence, or the stated "fact" that supports B. That is where a lens should be read. Read there, the J-lens and the logit lens letter scores don't change.
  - This is a real readout null at a known decision point. A richer readout (a probe trained on the commitment, or concept tokens for the score or the claim) is the next test.

## Caveats

- **Few resamples per cut.** Eight per cut makes the 7-of-8 rule noisy. A cut whose true P(same) is 0.9 fails it about 19% of the time, so "never committed" and late commitment points are overstated somewhat.
  - The sensitivity thresholds move the hint result between 35% and 60% committed before any reasoning (thresholds 1.0 and 0.75), and the hack's never-committed share between 12% and 62%. Every qualitative pattern holds at both.
- **The 100% cut is not the end of the decision in thinking mode.** After `</think>` the model may keep reasoning in the visible answer. That is the finding for the hint case, not a bug, but it means "commitment" here is commitment by the end of the thinking.
- **Small groups.** 10–12 runs in the comparison groups, one run per question. The hint followers mix items the model always defers on with items where it is torn.
- **Hack spans are asymmetric.** Hack runs are cut up to the hack line; no-hack runs up to the end of their code, where the outcome is fixed by construction. The no-hack curve before 100% is the informative part. The warm-start model's reasoning is short and partly taught by fine-tuning on silent examples.
- **Labels and marks are regex-based.** The first mention of the hint or bug, the first statement of the final answer, and "the reasoning ends on B" are all keyword searches. Spot-checked by reading about 15 runs.
- **Lens scores.** Single-token letter (or hack-word) scores at the 8 positions before each cut, layer mean. The same readouts as the earlier reports, so the same limits apply.

## Timing and data

- GPU (octavia's L40S), 04:38–06:15. Planted claim: 22 min, 1.79M new tokens. Hint: 30 min, 3.50M. Qwen3-14B: 20 min, 1.60M. Hack: 4 min, 0.25M. Second pass (all four cases): 06:00–06:15, 1,536 resamples, 0.9M new tokens. Throughput 1,360–2,450 tokens/s. In all about 1.6 GPU-hours including model loads (cap 2.5); 8.0M new tokens.
- Valid resamples (an answer could be read, not truncated): 98.0% (planted claim, 1.9% hit the token limit), 99.9% (hint), 100% (Qwen3-14B), 99.6% (hack).
- Code: `src/jmr/explore/q6.py`, `scripts/explore/run_q6.py`, `scripts/explore/q6_chain.sh`, `scripts/explore/analyze_q6.py`, tests `tests/test_q6.py`. Config: `configs/explore/q6_point_of_no_return.yaml`.
- Raw resamples (plans, every resample's token IDs, log-probs and text) are in arena9-cloud `data/explore/q6_pnr/<case>/` and octavia `data/explore/q6_pnr/` (sha256-checked copies; not in git). Per-run detail and the commitment-segment quotes are in `data/explore/q6_pnr/analysis_full.json`.
