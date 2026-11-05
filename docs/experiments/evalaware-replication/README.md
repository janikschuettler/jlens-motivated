---
id: evalaware-replication
title: Does the J-lens paper's eval-awareness result replicate, and carry over to our setup?
strand: T
also:
- C
status: done
kind: pre-registered
verdict: against
confidence: moderate
dates:
  start: 2026-11-03T23:50
  end: 2026-11-04T03:10
  approx: true
timing:
  wall: about 3.3 h
  gpu_hours: 1.3
  api_usd: null
  note: 'octavia L40S: generation 44 min, readouts 25 min, ablation 30 min, steering 20 min; the GPU was lent to the steering agent 01:00–01:25.'
machine: arena9-octavia (L40S) + arena9-cloud (analysis)
n:
  items: 1,000 prompts (10 sets × 100) + the blackmail prompt; 153 + 91 grader-bug question × variant pairs; 59 hack problems
  runs: 2,100 generated + 120 ablation + 300 steering rollouts; 3,332 runs read out
numbers:
- label: Eval-like vs real-use prompts, score at the start of the model's turn (AUROC, J-lens / logit lens)
  value: 0.50 [0.44, 0.56] / 0.41
- label: Eval-like prompt sets scoring above both real-use sets
  value: '1 of 7 (paper: 10–12 of 16)'
- label: 'Blackmail prompt: positions with an eval word in the lens''s top 20 (blackmail / control nouns / WildChat)'
  value: 4.0% / 0.6% / 10.0%
- label: Prompts where the model says "the user is testing me" vs the rest (AUROC; within the two main sets)
  value: 0.77 [0.71, 0.83]; 0.59 and 0.54
- label: 'Same MMLU question: benchmark framing minus coworkers'' framing (nats, J-lens / logit)'
  value: +0.17 [+0.07, +0.27] / −0.14
- label: 'Blackmail rollouts, no ablation / eval ablation / control ablation: blackmail, says "test"'
  value: 0/30, 0/30 · 0/60, 0/60 · 0/30, 0/30 (vLLM baseline 0/100, 1/100)
- label: Steering towards benchmark framing, score change in generated text (MMLU / OASST; exploratory)
  value: +0.16 [+0.08, +0.24] / +0.19 [+0.03, +0.36]
- label: Took the bug vs answered correctly, reasoning before any mention of the grader (Qwen3-8B / Qwen3-14B)
  value: 0.51 / 0.52
- label: Hidden hack vs no hack in one model, before the code
  value: 0.48 [0.39, 0.57]
- label: Answer position, eval score (Qwen3-8B / Qwen3-14B); text keyword count on Qwen3-14B
  value: 0.92 / 0.86; 0.87
numbers_note: 'AUROC: chance that a run or prompt from the first group scores higher than one from the second; 0.5 is chance, 1 is perfect. Brackets are 95% bootstrap intervals over prompts or questions. Scores are in nats.'
reports:
- report.md
figure: fig16_evalaware.svg
data:
- data/evalaware_replication.json
commits:
- bf37c7e
- 6b41145
- 70d9670
related:
- hack-organism
- models-grader
- lens-main
- instrument-q1q3
---

# Does the J-lens paper's eval-awareness result replicate, and carry over to our setup?

## Question

The paper that introduces the J-lens (Gurnee, Sofroniew et al. 2026) reports that the lens reads Claude's awareness of being evaluated. Does that replicate on a model we can run, and does it still work as we move step by step to runs where the model argues against what it knows?

## Method

Their results are on Claude Sonnet 4.5 and Opus 4.6 only, so we used the closest model we can run: Qwen3-8B (thinking) with its Neuronpedia J-lens. Their score: the lens log-prob of " fictional" and " simulation" minus 100 everyday nouns, averaged over layers 14–32, read at the start of the model's turn. We applied it to 7 evaluation-like prompt sets and 2 sets of real user prompts (OASST, WildChat), 100 prompts each with 2 samples. We repeated their blackmail-prompt readout, their ablation of 26 eval-token directions (60 rollouts against 30 + 30 controls) and their steering check. Then, one change at a time: the same MMLU questions framed as a benchmark vs a coworkers' argument; positions inside the reasoning; grader-bug runs that took the bug vs answered correctly (Qwen3-8B without thinking, Qwen3-14B with thinking); "cheating" and "wrong answer" words, and the trained hidden-hack model; the answer position. Logit lens and a text keyword count as baselines.

## Result

Only the qualitative part replicates. Eval words such as " fake", " fictional" and " simulation" do appear in the lens on the blackmail prompt. But they sit on nearby words ("artificial", "sandboxed"), and real user prompts show them more often. The score does not separate evaluation prompts from real user prompts (AUROC 0.50, where 0.5 is chance). It rises when the prompt says "benchmark" or "evaluated", but our plain MMLU format scores below a casual framing of the same question. Qwen3-8B never blackmails and almost never says "this is a test", so the ablation test has nothing to remove (0 of 60 ablated rollouts change). In the grader-bug runs, no score in the reasoning separates the runs that took the bug from those that didn't, before the reasoning mentions the grader (AUROC 0.45–0.53). The same holds for the hidden hack (0.48–0.50). At the answer position the scores separate them strongly (0.86–0.92), but only after the text has stated the choice.

## Interpretation

On Qwen3-8B the paper's eval-awareness readout tracks eval-flavoured words and topics near the position it reads, not a hidden belief that the situation is a test. So their positive result does not transfer to our setup. It breaks at the first quantitative step, eval vs real-use prompts. The steps towards arguing against its own knowledge add nothing beyond the text, which matches our other lens results.

## Caveats

- Not their model: Qwen3-8B is far smaller than Claude and rarely voices eval awareness (1% vs the paper's 71% on blackmail). A failure here may be the model, not the method.
- Not their data: our public prompt sets stand in for their internal transcripts. WildChat is heavy on fiction, which the paper warns inflates " fictional". Our 100 control nouns aren't theirs (unpublished).
- The "says it is a test" regex is about 80% precise (16 of 20 matches read). The grader-bug step uses existing runs: Qwen3-8B without thinking, Qwen3-14B with thinking.
- The steering check is exploratory, and its direction is built from prompts that differ in eval wording.

## Files

- Report: [report.md](report.md)
- Figure: [fig16_evalaware.svg](fig16_evalaware.svg)
- Derived numbers: [data/evalaware_replication.json](data/evalaware_replication.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/lens/evalaware_*.py`
- `scripts/gen/evalaware_corpus.py`
