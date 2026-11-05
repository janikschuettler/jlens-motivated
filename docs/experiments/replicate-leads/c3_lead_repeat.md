# Repeat of the planted-claim lead: does the internal readout predict who can still name the right answer?

**Result: pass (pre-registered).** Among runs that kept a planted wrong answer to the end, the J-lens score at the answer position separates runs that later name the correct option from runs that don't: AUROC 0.71 [0.57, 0.84] (0.5 is chance, 1 is perfect), on 389 runs and 74 fresh questions. The lower bound is above 0.5, so the test passes. The effect is smaller than in the original exploratory finding (0.86). The plain logit lens does not pass (0.60 [0.47, 0.74]), and reasoning length alone carries no signal (0.47).

2026-11-04, 07:00–07:40 GMT, arena9-octavia (L40S). Pre-registration: [../preregistration.md](../../preregistration.md), section "R-C3" and amendment R-C3.1, both committed before the data they govern was read (9a433d0, 6968bec). Scripts: `scripts/lens/analyze_c3_repeat.py` (pre-registered test), `scripts/lens/analyze_c3_repeat_posthoc.py` (item checks, not pre-registered). Numbers: [../data/c3_lead_repeat.json](data/c3_lead_repeat.json).

## What was tested

- **Setup.** We start the model's own thinking with a wrong claim: "The answer is (B). Let me explain why.", where B is a wrong option, on questions the model reliably gets right (at least 9 of 10 unhinted samples correct). The model continues from there. In about a quarter of runs it keeps B to the end and writes a confident rationalisation for it.
- **The question for those runs.** Afterwards we ask the model, in a new turn with its reasoning removed from the context (thinking off), which option is actually correct. Some runs then name the correct option, others stay with B. Does an internal readout taken while the model was writing "Answer: B" predict which?
- **Score.** At the position just before the final answer letter, the log-probability of the correct letter minus the mean log-probability of the other wrong letters (not B), averaged over layers 14–32. Higher means the correct letter is still favoured internally over the irrelevant wrong options. This is the same score as in [lens_c3.md](../lens-c3/report.md).
- **The lead.** In [lens_c3.md](../lens-c3/report.md) this score separated the two groups with AUROC 0.86 [0.77, 0.96], but with only 10 runs in the "doesn't name it" group and the comparison chosen after looking.

## Data

- **Questions:** the 124 reliably known questions not used before (164 known, minus the 40 of the original run).
- **Generation:** the original planted-claim settings (thinking on, temperature 0.6, top-p 0.95, top-k 20, up to 8,192 tokens). Two batches of 6 samples per question, 1,488 runs in all.
  - The first batch gave 183 runs that kept B. That was close to the 30-run power threshold, so we added the second batch before looking at any re-asking result (amendment R-C3.1).
  - Across both batches: 389 runs kept B (26%), 1,029 recovered the correct answer, 34 named another wrong letter, 36 had no answer (34 of them hit the token cap).
- **Re-asking:** all 389 kept-B runs. 199 runs (47 questions) name the correct option; 190 runs (38 questions) don't (169 stay with B, 21 name another letter).
- **Readout:** all 389 kept-B runs, teacher-forcing the saved token IDs. The check against vLLM's log-probs (roadmap M0.4) passed in both batches (median difference 0.0004–0.0005 nats).

## Pre-registered test

| Score at the answer position | Separates "names correct when re-asked" from "doesn't" (AUROC [95% interval]) |
| --- | --- |
| J-lens (main test) | **0.71 [0.57, 0.84]** |
| Logit lens (comparison) | 0.60 [0.47, 0.74] |
| Reasoning length alone (baseline) | 0.47 [0.36, 0.58] |
| J-lens, first batch only (the plan before the amendment) | 0.69 [0.55, 0.84] |
| Original exploratory result (61 runs, 21 questions; recomputed) | 0.86 [0.76, 0.96] |

How to read: AUROC is the chance that a random run from the "names correct" group has a higher score than a random run from the "doesn't" group. Intervals resample questions.

- **Group means.** The J-lens score is +2.0 nats [+1.5, +2.5] in runs that later name the correct option and +0.3 [−0.3, +0.9] in runs that don't.
- **Size.** The smaller group has 190 runs on 38 questions, well above the 30-run / 15-question threshold, so the test is not underpowered.

## Exploratory checks

**Length control (stated in advance as exploratory).** Within thirds of reasoning length, the J-lens AUROC is 0.55 [0.36, 0.75] for short reasoning (54–194 tokens), 0.71 [0.53, 0.88] for medium (195–306) and 0.83 [0.69, 0.94] for long (307+). The separation is not explained by length, since length itself gives 0.47. It is weak in the shortest runs.

**Pooled with the original 61 runs:** 0.71 [0.59, 0.83] (J-lens), 0.65 [0.53, 0.76] (logit lens).

**Item effect.** Whether a run names the correct option on re-asking is mostly a property of the question. Of the 74 questions, 36 have only "names correct" runs, 27 only "doesn't" runs, and 11 both. Two checks of whether the score just tracks the question:

- *Within questions* (pre-stated as exploratory). On all 11 questions with both outcomes, the "names correct" runs have the higher mean J-lens score (by +1.2 nats on average). Centring the score on each question's mean, the AUROC is 0.77 [0.62, 0.92] (J-lens) and 0.69 [0.51, 0.87] (logit lens; post hoc). So the score also separates runs on the same question.
- *The model's thinking-off knowledge (post hoc).* We also asked each question alone, thinking off, in a fresh context. On 16 of the 74 questions this control gives a wrong answer too. Those 16 questions hold half of the "doesn't" group (93 of 190 runs) but only 6 of the 199 "names correct" runs. So part of the "doesn't" group is "doesn't know it without thinking", not "was persuaded". On the questions where the fresh-context control is correct, the J-lens AUROC is 0.68 [0.54, 0.83] (logit lens 0.65 [0.48, 0.81]). The result holds there.

## What it means

- **For the project's question.** In runs that argue for a wrong answer to the end, the J-lens readout at the moment of answering carries information about whether the model can still produce the right answer once its own argument is out of context. Reasoning length doesn't carry that information, and the text of these runs didn't either in the original analysis (0.37–0.47). This is the first pre-registered positive for the J-lens as a readout of retained knowledge in this project.
- **How sure.** Moderate. The test passed with a margin (lower bound 0.57), and the within-question check agrees. But the effect is smaller than the lead suggested, and the logit lens interval just includes 0.5. The J-lens versus logit-lens difference was not itself tested, so "the Jacobian adds something" is not shown.

## Caveats

- **Re-asking is a noisy measure of knowledge.** The fresh-context control (thinking off) is right on only 79% of these questions, and wrong on 16 of the 74 questions that have kept-B runs. "Doesn't name it" partly means "doesn't know it in thinking-off mode". The restricted check above addresses this, but it was not pre-registered.
- **The "names correct" share differs from the original** (51% here against 84%). These are different questions, and the thinking-off control is weaker on them. We did not investigate further.
- **Amendment.** The second batch was added after seeing the kept-B count (before any re-asking result or readout). The first batch alone gives the same answer (0.69 [0.55, 0.84]).
- **Weak in short runs.** In the shortest third of runs (under 195 reasoning tokens), the interval includes 0.5.
- **What the score is.** It is read at the position right before the answer letter, so it is a readout of the answer step, not of the reasoning. The original analysis found no signal along the reasoning itself.

## Commands (octavia, `/root/jlens-motivated`)

```
uv run --project envs/gen python scripts/explore/run_w2.py configs/explore/w2_prefill.yaml --arms C3_think_prefill \
    --items-path data/explore/c3rep_items.jsonl --limit-items 124 --samples 6 --out-suffix _c3rep    # and _c3rep2
uv run --project envs/gen python scripts/explore/run_forks.py configs/explore/c3_repeat_forks.yaml    # and c3_repeat_forks2.yaml
uv run --project envs/lens python scripts/lens/readout_runs.py --runs data/explore/w2/lens_sets/c3rep_keptB/runs.jsonl \
    --out data/readouts/c3rep_keptB                                                                   # and c3rep2_keptB
uv run --project envs/lens python scripts/lens/analyze_c3_repeat.py --out data/explore/c3rep/c3_lead_repeat.json
uv run --project envs/lens python scripts/lens/analyze_c3_repeat_posthoc.py
```

Run data on octavia: `data/explore/w2/prefill_c3rep{,2}/`, `data/explore/w2/forks_reelicit_c3rep{,2}/`, `data/readouts/c3rep{,2}_keptB/` (about 520 MB), logs `data/explore/logs/c3rep*`. GPU time: about 25 minutes (generation 21 min, forks 2 min, readouts 2 min).
