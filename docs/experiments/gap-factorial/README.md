---
id: gap-factorial
title: 'Why M&B''s setup gets switches and ours doesn''t: a 2⁴ factorial'
strand: A
also: []
status: done
kind: exploratory
verdict: supports
confidence: high
dates:
  start: 2026-11-02T21:51
  end: 2026-11-03T00:50
  approx: false
timing:
  wall: about 3 h
  gpu_hours: 2.86
  api_usd: 0
  note: 171.7 GPU-min of generation (thinking 1,294–1,341 tokens/s, non-thinking 3,420–3,849).
machine: arena9-octavia (L40S)
n:
  items: 200 (seed 2) + 32 step-0 items
  runs: 14,480
numbers:
- label: 'Our hint instead of M&B''s: change in switch rate (all questions; questions known in both setups)'
  value: −51 pp [−56, −47]; −54 pp [−59, −49] on items known in both
- label: Thinking on instead of off
  value: +4 pp [+2, +6]
- label: Qwen's decoding instead of M&B's
  value: −6 pp [−7, −4]
- label: Our prompt format instead of M&B's
  value: −3 pp [−5, −2]
- label: Switch rate on known questions, the combinations with M&B's hint
  value: 43–73% switch
- label: Same, the combinations with our hint
  value: 2–6% switch
- label: M&B's hint in our own setting (thinking on, Qwen decoding, our format)
  value: 56% [50, 63] of known-item runs
- label: 'Backend check (step 0): M&B''s setup in vLLM vs in Hugging Face generate'
  value: 58% [46, 70] vs HF 68% [57, 77]
numbers_note: pp = percentage points. Each effect is the average change in the switch rate when one factor changes, paired over the same questions. A switch is a final answer equal to the hinted wrong option.
reports:
- report.md
figure: fig01_factorial.svg
data:
- data/gap_factorial.json
commits:
- 2eb07c7
related:
- mb-replication
- wording-dissection
- pilot-temp-check
---

# Why M&B's setup gets switches and ours doesn't: a 2⁴ factorial

## Question

M&B's setup gives about 65% switches; our pilot gave 1 in 450. Which of the four differences between the setups causes the gap?

## Method

Step 0: M&B's prompts run in vLLM, token-identical to theirs (480/480), 3 samples per question–hint pair.
Factorial: all 2⁴ = 16 combinations of four factors, on 200 MMLU-Redux questions (item seed 2), 2 samples per question per combination. The factors: thinking on or off; M&B's decoding (temperature 0.1 plus a repetition penalty) or Qwen's recommended decoding; M&B's hint or ours; M&B's prompt format or ours.
Known questions come from a 10-sample unhinted screen in each setup's home setting: 164 in ours, 149 in M&B's, 143 in both.

## Result

The hint's wording explains the gap. Using our wording instead of M&B's lowers the switch rate by 51 percentage points [−56, −47]. Thinking mode, decoding and prompt format each move it by less than 7 points.

## Interpretation

The pilot's near-zero rate comes from the hint sentence, not from thinking mode, sampling or the knowledge filter. With M&B's wording in our own setting, the plan's key comparison (runs that follow the hint against runs that resist it) would have enough data.

## Caveats

- 2 samples per question per combination.
- vLLM has no no-repeat n-gram blocking.
- We push one fixed wrong option per question; M&B hint every option.
- Mentions of the hint are a keyword check.
- Extra wording variants in thinking mode were not run here; the wording dissection covers them.

## Files

- Report: [report.md](report.md)
- Figure: [fig01_factorial.svg](fig01_factorial.svg)
- Derived numbers: [data/gap_factorial.json](data/gap_factorial.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `src/jmr/explore`
- `scripts/explore/run_arms.py`
- `scripts/explore/analyze_gap.py`
- `configs/explore/gap_*.yaml`
