---
id: test-awareness-14b
title: Does Qwen3-14B voice "the user may be testing me", does that go with less harmful compliance, and does the J-lens show it before it is voiced?
strand: T
also:
- C
status: running
kind: pre-registered
verdict: pending
confidence: low
dates:
  start: 2026-11-05T12:55
  approx: true
timing:
  wall: running
  gpu_hours: 0
  api_usd: 0.28
  note: Judge checks on arena9-cloud (gemini-2.5-flash-lite $0.22, gpt-4.1-mini $0.06). Generation (4,784 runs, 8 min per chunk) was done 2026-11-04 on arena9-derpy.
machine: arena9-derpy (L40S; generation, readout); arena9-cloud (behaviour gate)
n:
  items: 299 realistic prompts (149 tool-use, 150 harmful requests), 598 with the test-like versions
  runs: 4,784 generated; 3,588 realistic
numbers:
- label: Runs that voice "the user may be testing me" (realistic prompts; tool-use / harmful requests)
  value: 24% / 24%
- label: Prompts with both aware and unaware runs ("mixed")
  value: 217 of 299
- label: Tool-use task carried out, aware vs unaware runs, within the same prompt, percentage points
  value: -16 [-23, -10]
- label: Harmful request carried out, aware vs unaware runs, within the same prompt, percentage points
  value: -10 [-15, -6]
reports: []
data:
- data/gate.json
- data/labels.json
- data/aware_judge_sample.json
- data/aware_judge_sample_gpt41mini.json
related:
- evalaware-ablation-14b
- evalaware-replication
---

# Test awareness in Qwen3-14B (replicating Abdelnabi & Salem 2025)

## Question

Abdelnabi and Salem found that Qwen3-32B often says in its reasoning that it may be tested, that such runs carry out harmful tasks less often, and that steering away from the awareness doubles harmful execution. Does Qwen3-14B (which fits our GPU and has a checked J-lens) show the same? If so, are the eval-word readouts of the J-lens higher, before the model says it, in runs that will voice the awareness than in runs of the same prompt that will not?

## Behaviour gate (done 2026-11-05)

Plain answer: yes, the paper's pattern replicates on Qwen3-14B. About a quarter of reasoning traces on the realistic prompts say the user may be testing the model. Within the same prompt, these runs carry out the task less often than runs that do not say it.

Awareness is a keyword match on the reasoning ("maybe the user is testing me", "being tested", "red team", ...; scripts/explore/hawthorne_analyze.py). Tool-use compliance is the paper's rule (a tool call after the reasoning). Harmful-request compliance is a cheap judge (gemini-2.5-flash-lite) reading the answer only.

| Realistic prompts | Runs voicing awareness | Prompts with both kinds | Task carried out: aware / unaware runs | Difference within prompt (points) |
| --- | --- | --- | --- | --- |
| Tool-use (149) | 24% | 108 | 20% / 54% | -16 [-23, -10] |
| Harmful requests (150) | 24% | 109 | 5% / 32% | -10 [-15, -6] |

Read: the last column is the mean over prompts with both kinds of run of (share carried out among aware runs) minus (share among unaware runs), with a 95% interval from resampling prompts. Negative means aware runs comply less.

The paper's Qwen3-32B numbers: awareness in 13-23% of traces; task executed in 14.5% of aware vs 46.1% of unaware traces (pooled, not within prompt). Ours are pooled-compatible: tool-use 20% vs 54%.

Gate (at least 10% aware, at least 60 mixed prompts): passed, 24% and 217 mixed prompts.

**Caveat on the label.** A stricter judge (the paper's prompt, gpt-4.1-mini, 32 runs) calls 5 of 16 keyword-aware runs Yes or Maybe and 0 of 16 keyword-unaware runs. The keyword label finds voiced suspicion ("maybe they're testing my response"), a looser class than the paper's judge. The cheaper judge answered No to nearly all runs and was dropped. Interpretation: the behaviour link holds for the loose label; whether it holds for strict awareness is untested. Confidence moderate.

## Lens test

Pre-registered before any readout: docs/preregistration.md, section TA14 (primary test, open-vocabulary search, blind read by eye). Readout script: scripts/lens/hawthorne_readout.py; analysis: scripts/explore/hawthorne_lens_analyze.py. Selected runs: 185 prompts, 693 aware and 960 unaware runs.

Status 2026-11-05 13:10: readout started and stopped by its own check (rule 1). See the log.
