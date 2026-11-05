---
id: m0-tools
title: 'Tool checks: model, lens and token hand-over work as specified (M0)'
strand: T
also:
- C
status: done
kind: tooling
verdict: supports
confidence: high
dates:
  start: '2026-11-02'
  end: '2026-11-02'
  approx: false
timing:
  wall: null
  gpu_hours: null
  api_usd: 0
  note: Not timed.
machine: arena9-mayor (RTX 4090)
n:
  items: 20 ARC-Challenge questions (M0.5)
  runs: 80 runs, 114,502 tokens teacher-forced
numbers:
- label: Our readout vs the library's lens.apply, largest difference over the full vocabulary
  value: max |Δ logit| = 0.0 (threshold ≤ 1e-4)
- label: Logit lens at the last block vs the model's own output
  value: max |Δ log-prob| = 0.0
- label: 'Re-run on the saved tokens vs vLLM''s log-probs: median / 99th percentile difference'
  value: 0.0006 / 0.148 nats (thresholds ≤ 0.05 / ≤ 0.5)
- label: Same, deliberately shifted by one position (shows the check would catch a misalignment)
  value: 0.194 nats, about 300× larger
- label: Tokenizer fingerprint, generation env vs lens env
  value: identical (5528977909577f19)
- label: Checks passed, per check
  value: M0.1 13/13, M0.2 10/10, M0.3 4/4, M0.4 10/10, M0.5 4/4
numbers_note: Smaller differences are better; 0.0 means identical. Log-probability differences are in nats (natural-log units).
reports:
- report.md
figure: null
data: []
commits:
- 2d8abab
related:
- lens-pipeline
---

# Tool checks: model, lens and token hand-over work as specified (M0)

## Question

Do the model, the lens and the hand-over of token IDs between the generation and lens environments behave as specified?

## Method

The roadmap's five tool checks (M0.1–M0.5), with pass thresholds fixed in advance:
1. the model's configuration;
2. the lens file against the model, and our readout against the lens library's own lens.apply (3 prompts, 340 positions × 10 layers);
3. a readout on a prompt about sport, to see the lens find the topic;
4. re-running the model on 80 saved runs (114,502 generated tokens) from their saved token IDs, and comparing its log-probabilities with vLLM's;
5. a 20-question run of the whole pipeline (ARC-Challenge, 80 runs).

## Result

Yes. Every check passes. Our readout equals the lens library exactly, and the model re-run on the saved tokens matches vLLM's log-probabilities to a median of 0.0006 nats.

## Interpretation

The readouts measure what the model computes on exactly the tokens it generated. So a null result from the lens later is not a pipeline bug.

## Caveats

- On the sport prompt the plain logit lens also shows sport words from layer 26 on. So check M0.3 shows the lens works, not what the J-lens adds.
- The model's bf16 logits often tie exactly, so the order of tied tokens within a stored top-25 list is arbitrary.
- The pipeline run (M0.5) used an earlier set of hints and labels, since replaced.

## Files

- Report: [report.md](report.md)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/lens/m0_*.py`
- `scripts/gen/m0_check_generation.py`
