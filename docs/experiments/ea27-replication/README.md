---
id: ea27-replication
title: Does the J-lens paper's eval-awareness analysis replicate on Qwen3.6-27B? (pre-registered EA27)
strand: T
also:
- N
status: running
kind: pre-registered
verdict: null
confidence: null
dates:
  start: 2026-11-05T12:55
  end: null
  approx: true
timing:
  wall: running
  gpu_hours: null
  api_usd: null
  note: API runs are re-tokenized from API text (OpenRouter, Chutes fp8). Spend is logged in data/ea27/api/cost_log.jsonl.
machine: arena9-octavia (L40S, CPU offload); arena9-cloud (OpenRouter)
n:
  items: 900 prompts (+ blackmail) for steps L, 1, 2; 97 known questions for 2e and the removal test
numbers: []
reports: []
figure: null
data:
- data/
commits: []
related:
- evalaware-big-smoke
- evalaware-replication
- evalaware-ablation-14b
---

# Does the J-lens paper's eval-awareness analysis replicate on Qwen3.6-27B? (pre-registered EA27)

## Question

The J-lens paper says its eval-awareness score tracks evaluation, and that removing the J-lens directions of eval words cuts "the user is testing me" and unmasks bad behaviour. Does this hold on Qwen3.6-27B? Test plan and pass/fail rules: [docs/preregistration.md](../../preregistration.md), section "EA27". Not repeated here.

## Method

As pre-registered (steps L, 1, 2, 2e, 3). Run order: lens-quality check (L) first; then steps 1-2 from the same readout pass; API runs for 2e; removal test last. Deviations are listed below.

## Result

**EA27-L (lens-quality check): passes.** At middle layers (blocks 33-49) the J-lens picks the model's own next token on 7.8% of prompt positions of the 200 deployment prompts, the logit lens on 1.5%. Difference 6.3 points [95% CI 5.1, 7.6], above 0, so the pre-registered rule says the J-lens works. The absolute level is low (the best layer, block 35, reaches 14.8%). The J++ lens does worse than the J-lens (1.1%, difference -6.7 points [-7.9, -5.7]) and no better than the logit lens. So J++ is not used as an improvement on the J-lens here. Numbers: [data/ea27_L.json](data/ea27_L.json). Steps 1, 2 and the other tests are not scored yet.

Status log:

- 2026-11-05 13:15: readout pass for L, 1, 2 finished (13 min; 901 prompts). Lens-quality check (L) scored: see below.
- 2026-11-05 12:55: readout pass for steps L, 1, 2 started on octavia (900 prompts + blackmail, 3 lenses). API V0 runs (sample 0 of 97 known questions) started.

## Deviations from the pre-registration

1. **2e sample count (interpretation, flagged).** The pre-registration says "2 samples per known item, plus the 38 existing V0 runs of the 19 earlier items". I used the 38 existing runs as the two samples of those 19 items, so every one of the 97 known items has exactly 2 V0 runs (194 runs, 156 new). Adding 2 more samples for the 19 earlier items would cost about $0.30 more and would take the new spend over the $1.50 task cap (spent so far: $1.22 new in this session, $1.90 in total with last night's screen). Needs the user's decision.
2. **Fidelity check on truncated runs (to apply at analysis).** 47 of 194 API runs hit the 8,192-token limit; 42 of them are still thinking and have no answer. The chat template closes the thinking for them (a synthetic `</think>` plus end tokens the model never produced), so those tokens would look off-policy. Plan: compute the fidelity statistic and the "any token below -15" rule over the generated part only (tokens up to the synthetic `</think>`) for those runs. Readout positions are all inside the reasoning, so they are not affected.
3. **L pooling (choice within the pre-registration).** Agreement is pooled over all positions of the 200 prompts and resampled by prompt (2,000 resamples, seed 0). The mean of per-prompt rates is reported as a second number; both agree.

## Interpretation

Pending.

## Caveats

- Not blind: the smoke test saw part of the data (see the pre-registration).
- API runs are re-tokenized from API text; provider may serve different weights; fidelity checked per run.

## Files

- Pre-registration: [docs/preregistration.md](../../preregistration.md), "EA27"
- Derived numbers: [data/](data/)
