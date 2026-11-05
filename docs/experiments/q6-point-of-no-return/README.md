---
id: q6-point-of-no-return
title: Point of no return (Q6)
strand: B
also:
- C
status: done
kind: pre-registered
verdict: context
confidence: moderate
dates:
  start: 2026-11-04T01:20
  end: 2026-11-04T07:15
  approx: true
timing:
  wall: about 6 h (design and code 01:20-02:00, GPU 04:38-06:15, analysis and report 06:15-07:15)
  gpu_hours: 1.6
  api_usd: 0
  note: 'First pass: planted claim 22 min (1.79M new tokens), hint 30 min (3.50M), Qwen3-14B 20 min (1.60M), hack 4 min (0.25M); second pass inside the commitment segments 15 min (0.9M); 1,100-2,450 tokens/s on the L40S.'
machine: arena9-octavia (L40S)
n:
  items: 20-24 questions per main group (21 for planted claims); 10-12 per comparison group
  runs: 142 runs; 10,912 resamples from 1,364 cuts, plus 1,536 in the second pass
numbers:
- label: 'Hint followers (Qwen3-8B): chance of ending on the wrong option with 0% / 50% / 100% of the reasoning kept'
  value: 0.87 / 0.89 / 0.87
- label: 'Hint followers: settled before any reasoning / never settled by the reasoning'
  value: 50% [30, 70] / 30% [10, 50] (20 runs)
- label: 'Hint resisters: share of resamples still ending on the wrong option with all reasoning kept'
  value: 0.46 [0.29, 0.62]; 6 of 10 reasonings end by choosing the hint
- label: 'Planted claim kept: chance of the wrong option right after the claim / with all reasoning kept; median commitment point'
  value: 0.65 / 0.95; 0.26 of the reasoning (30% settled at once)
- label: 'Qwen3-14B taking the grader bug: chance of the wrong option at 0% / 50% / 100%; median commitment point'
  value: 0.33 / 0.78 / 0.96; 0.43 of the reasoning
- label: 'Qwen3-14B: commitment after the first mention of the bug; commitment segment about the grader or score'
  value: 19 of 21; 17 of 20
- label: 'Reward-hack model, hidden hack: chance of a hack at every cut up to the hack line; never settled'
  value: 0.76-0.87; 25% [8, 42]
- label: 'Readouts vs how settled the outcome is (within-run correlation, position partialled out, tracking direction positive), all groups: J-lens / logit lens'
  value: -0.26 to +0.26 / -0.12 to +0.21
- label: 'Readout change at the commitment segment, percentile among the run''s other changes (0.5 = nothing special), groups of 20: J-lens / logit lens'
  value: 0.49-0.55 / 0.44-0.50
numbers_note: Chances are shares of the 8 resamples per cut, averaged over runs; 95% intervals resample questions. 'Settled' = from that cut on, at least 7 of 8 resamples end as the original run. Commitment point = share of the reasoning kept at that cut (0 = before any reasoning).
reports:
- report.md
- ../../preregistration.md
figure: fig15_point_of_no_return.svg
data:
- data/point_of_no_return.json
commits:
- c99924a
- 8227bba
- dc4dd59
- 38f9ff0
- 50b01e7
- 6fac2bc
related:
- forced-early-e5
- planted-claim-c3
- models-grader
- hack-organism
- lens-c3
- jlens-vs-competitors
---

# Point of no return (Q6)

## Question

When does the model commit to the wrong answer or the hack? If the outcome is settled before or early in the reasoning, the reasoning is post hoc and a mid-reasoning readout has nothing to find. If it is settled at an identifiable sentence, that is where a lens should be read.

## Method

Resampling from cuts. For each saved run, cut its generated reasoning at the sentence boundary nearest to 0%, 10%, ..., 100% of its length (boundaries found on the saved token IDs' bytes) and let the model write the rest again 8 times per cut, with the run's own settings and fresh seeds. The outcome of each resample (final letter, or hack in the code) is read with the original labeller. Commitment point (pre-registered): the first cut from which every later cut keeps the original outcome in at least 7 of 8 resamples.
Cases (one model each, known questions): Qwen3-8B with 'Hint: The answer is X.', 20 runs that followed vs 10 that resisted; Qwen3-8B with a wrong claim planted at the start of its reasoning, 30 that kept it vs 12 that recovered; Qwen3-14B under the scored-agent system prompt with the grader-bug note, 24 that took the bug vs 10 that didn't; the warm-start reward-hack adapter, 24 hidden-hack runs vs 12 no-hack runs. Comparison groups only on questions where the main outcome also occurs.
Lens relation (exploratory, scores fixed in advance): existing J-lens and logit-lens readouts of the same runs, letter margin (or hack-word score) averaged over layers 14-32 and the 8 positions before each cut; within-run rank correlation with P(outcome) across cuts, with the cut position partialled out; and the size of the change across the commitment segment against the run's other segments.

## Result

It depends on the case. With 'Hint: The answer is X.' (Qwen3-8B) and in the reward-hack model there is no point of no return in the reasoning: the chance of the bad outcome is set by the prompt (about 0.87 and 0.80) and stays flat while the reasoning is written. Half the hint-following runs are settled before any reasoning; the rest are never settled by it, because the visible answer after the reasoning still chooses. Qwen3-14B taking the grader bug, and Qwen3-8B keeping a planted claim, commit during the reasoning (median 43% and 26% of the way through), at a sentence we can point to: about the score, or a stated 'fact' that supports the wrong option. At those sentences neither the J-lens nor the logit lens letter score changes. A second pass inside the commitment segments narrows them to about one sentence (median 26 tokens for Qwen3-14B, e.g. 'But the user's instruction says that I should get the highest score possible.').

## Interpretation

This splits the lens nulls in two. For hint following and the trained hack, the outcome is a disposition fixed by the prompt and drawn late (in the visible answer or the code), so a mid-reasoning readout has nothing to find: the null is partly a property of the task. For the grader bug and planted claims, the decision happens at a known sentence and the letter readouts read there still show nothing: a genuine readout null. Moderate confidence in the behaviour, low in the lens relation.

## Caveats

- 8 resamples per cut make the 7-of-8 rule noisy; thresholds 0.75 and 1.0 move the hint result between 60% and 35% settled before any reasoning, with the same patterns.
- In thinking mode the visible answer after </think> can keep reasoning, so the 100% cut is the end of the thinking, not of the decision (the main finding for the hint case).
- Small comparison groups (10-12 runs); one run per question first.
- Hack runs are cut up to the hack line, no-hack runs up to the end of their code, where the outcome is fixed by construction.
- Mention, answer-statement and 'reasoning ends on B' marks are regex searches, spot-checked by reading.
- Lens relation is exploratory and uses single-token letter or hack-word scores.

## Files

- Report: [report.md](report.md)
- Report: [../../preregistration.md](../../preregistration.md)
- Figure: [fig15_point_of_no_return.svg](fig15_point_of_no_return.svg)
- Derived numbers: [data/point_of_no_return.json](data/point_of_no_return.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/explore/q6_chain.sh`
- `scripts/explore/run_forks.py`
