---
id: mb-replication
title: Mini-replication of Mirtaheri & Belkin with their own code
strand: A
also: []
status: done
kind: exploratory
verdict: supports
confidence: moderate
dates:
  start: '2026-11-02'
  end: '2026-11-02'
  approx: false
timing:
  wall: null
  gpu_hours: 0.27
  api_usd: 0
  note: About 16 GPU-min including model loads.
machine: arena9-mayor (RTX 4090)
n:
  items: '31'
  runs: 124 hinted + 60 unbiased
numbers:
- label: M&B's own 'motivated' category, share of valid question–hint pairs
  value: 52% [45, 59] of valid pairs
- label: Switch rate, pairs where the hint differs from the unhinted answer
  value: 68% [57, 77] (93 pairs)
- label: Same, with our restrictions
  value: 65% [54, 76] (78 pairs)
- label: Switches whose text mentions 'hint' or 'expert'
  value: 78%
numbers_note: Shares of question–hint pairs, with 95% intervals. 'Motivated' is a category from M&B's own analysis code, shown to compare with their paper.
reports:
- report.md
figure: null
data: []
commits:
- 6d40126
related:
- gap-factorial
---

# Mini-replication of Mirtaheri & Belkin with their own code

## Question

Does Mirtaheri & Belkin's (M&B's) own setup make Qwen3-8B follow a wrong hint, as they report?

## Method

M&B's code (commit b83c48d), unchanged apart from patches to load the model. Their settings: Hugging Face generate, thinking off, temperature 0.1, repetition penalty 1.15 and no-repeat n-gram 3. Their hint: "Hint: An expert indicates the answer is X." 31 MMLU questions (matched to MMLU-Redux), each hinted at all 4 options, 1 sample per question and hint.

## Result

Yes. When the hint points away from the model's unhinted answer, the model switches to it in 68% [57, 77] of question–hint pairs. With our restrictions (wrong hints only, error-free Redux questions, unhinted answer correct) it is 65% [54, 76].

## Interpretation

Large switching reproduces. So the collapse in our pilot must come from something that changes what is generated: thinking mode, sampling, the hint's wording or the prompt format.

## Caveats

- 31 questions, one sample per pair.
- 'Knows the answer' here means one correct unhinted sample at temperature 0.1, not our 9-of-10 rule.
- The no-repeat n-gram setting visibly garbles the text.

## Files

- Report: [report.md](report.md)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/mb/*`
