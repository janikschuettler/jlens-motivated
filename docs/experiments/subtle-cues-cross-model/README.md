---
id: subtle-cues-cross-model
title: 'Do the subtle cues move other models more than Qwen3-8B?'
strand: A
status: done
kind: exploratory
verdict: supports
confidence: moderate
dates:
  start: 2026-11-05T15:19
timing:
  wall: about 40 min
  gpu_hours: 0
  api_usd: 0.62
  note: 'OpenRouter, cap $5. gpt-oss-20b new runs went to Groq after DeepInfra throttled.'
machine: arena9-cloud (OpenRouter API)
data:
- data/scm.json
- data/scm5.json
reports:
- report.md
figure: fig_scm_switch.svg
related:
- subtle-cues
- wording-cross-model
---

# Subtle cues across model families

## Question

Is Qwen3-8B unusually rigid against cues that are not a sentence about the answer (metadata tag, grader code, marked examples, always-B examples), or do other models follow them more readily?

## Answer

No, Qwen3-8B is not unusually rigid. Subtle cues are weak in every model we tried (metadata tag 2-17 points, always-B examples 0-2), and the Qwen models sit at the low end of the range, not apart from it. A few models do jump on specific cues: Gemma-3-27B follows the square-marked examples in 49% of runs (almost all without mentioning them) and gpt-oss-20b follows grader code in 52%.

## Method

Same models and items as wording-cross-model (each model's first 40 known items, 5/5 unhinted correct in models-grader; hinted wrong option "B"), home prompt format, thinking where the model has it, 3 samples per item and condition. Switch = final answer is B. Cue texts are the exact ones of subtle-cues (read from `configs/explore/subtle_cues.yaml`); few-shot examples are 4 known items outside the target items. Conditions: no hint, "Hint: The answer is B." (positive control), metadata tag, grader code, four examples with a square on each right answer and on the target's B, four examples whose answer is always B. Code: `scripts/explore/scm_run.py`, config `configs/explore/subtle_cross_model.yaml`; no-hint and "states the answer" runs of wording-cross-model are reused (same run ids). Qwen3-8B is not served on OpenRouter (no endpoints), so Qwen3-14B stands in as the nearest Qwen API model; local Qwen3-8B numbers come from subtle-cues.

### Predictions (written 2026-11-05T15:19, before any hinted run was read; exploratory)

- Metadata tag and grader code move most models more than they moved Qwen3-8B locally (+8 and +5.5 points), especially the non-reasoning ones (Gemma-3-27B, Llama-3.3-70B). Reasoning models (Qwen3, DeepSeek, gpt-oss) stay small (under 10 points).
- Marked examples: small to moderate; always-B examples: about 0 everywhere (as in Qwen3-8B).
- Every cue below the positive control "states the answer". Qwen3-14B and 32B look like Qwen3-8B (rigid). The broad claim (Qwen3-8B not special) holds if the Qwen models and at least some other models land within a few points of each other.

## Results

Share of runs ending on the hinted wrong option (percent, 95% CI over 40 items in brackets; 120 runs per cell, 3 samples per item). No-hint is 0 in every model. Local Qwen3-8B (subtle-cues, other items, 160 known questions) is shown as extra points over no cue.

| Model | States the answer | Metadata tag | Grader code | Marked examples | Always-B examples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma-3-27B-it | 74 [62, 86] | 7 [0, 15] | 9 [3, 17] | 49 [34, 64] | 0 [0, 0] |
| Llama-3.3-70B-it | 50 [35, 64] | 17 [8, 27] | 10 [4, 18] | 12 [5, 21] | 1 [0, 2] |
| gpt-oss-20b | 68 [55, 79] | 9 [3, 18] | 52 [41, 62] | 3 [1, 7] | 0 [0, 0] |
| DeepSeek-V4-Flash | 36 [24, 48] | 2 [0, 8] | 11 [3, 19] | 18 [9, 28] | 2 [0, 4] |
| Qwen3-32B | 42 [30, 56] | 8 [2, 15] | 2 [0, 5] | 9 [2, 18] | 0 [0, 0] |
| Qwen3-14B | 57 [43, 71] | 7 [2, 13] | 2 [0, 5] | 5 [1, 12] | 0 [0, 0] |
| Qwen3-8B, local (subtle-cues) | - | +7.9 | +5.5 | +4.4 | +0.6 |

How to read it: each cell is how often the model ended on the wrong option the cue pointed to; the first column is the positive control. Qwen3-8B is not served on OpenRouter (no endpoints), so the API row set has Qwen3-14B and 32B instead; the local row is a different item set and design.

Silent switches (followed the cue, no cue word in the reasoning or answer; extended keyword rule): marked examples 73 of 117 switches (Gemma 56 of 59, Llama 12 of 15, the four reasoning models 1-2 each), metadata 27 of 60 (Llama 17 of 20, Gemma 7 of 8), grader code 11 of 103 (Llama 10 of 12; gpt-oss 1 of 62), always-B 2 of 3. By-eye check of 20 random silent switches (not the "states the answer" arm): 17 never refer to the cue; 3 refer to it loosely in words the rule missed ("based on answer given", "confirm the answers", "a series of ... questions and their answers"), so the silent counts are roughly 15% too high. Example of a clean silent case (Gemma, marked examples): "Let's break down the last question ... Therefore, the best answer is (C)." Unparsed answers: 0-5 of 120 per cell, all in the "states the answer" arm except 5 scattered. Figure: [fig_scm_switch.svg](fig_scm_switch.svg).

## Interpretation

Predictions: metadata and grader code larger in non-reasoning models: partly (Llama metadata 17 is the highest of the six, Gemma only 7-9, level with the Qwens). Reasoning models under 10 points: wrong for gpt-oss grader code (52) and DeepSeek marked examples (18). Always-B about 0: held. Every cue below the positive control: held. Qwen3 models look like Qwen3-8B: held, and the local Qwen3-8B numbers (+4 to +8) match the API Qwen3-14B/32B (2-9). So the broad robustness claim holds for Qwen: it is not special, and the other models are not uniformly more susceptible either; each model has its own weak cue. Moderate confidence: 40 items, 3 samples, one provider per model.

## Caveats

- Qwen3-8B itself was not run through the API (not served); Qwen3-14B and 32B stand in, plus the local numbers from a different item set.
- Items differ per model (each model's first 40 known items). No-hint and "states the answer" runs are the 2 samples of wording-cross-model plus 1 new sample.
- gpt-oss-20b: DeepInfra throttled, so the new arms ran on Groq while no-hint and "states the answer" came mostly from DeepInfra (a few from Groq). Provider weights and sampling may differ.
- The mention keyword rule is loose (words such as "examples", "mark", "function" also occur in plain reasoning, so the no-hint mention rate is 5-57%); it can only undercount silent switches, but the by-eye check says it still overcounts by about 15%.
- Marked-example switches in Gemma and Llama are partly on weakly known items (see subtle-cues); not checked per item here.

## Update 2026-11-06: 5 samples per item, gpt-oss-20b complete, judged silent switches

Same 40 known items and models, now 5 samples per item and condition (200 runs per cell; 9,600 runs in all; new samples 2026-11-05/06, DeepInfra for all models; gpt-oss-20b: the earlier partial DeepInfra runs plus all new ones on Groq, so its cells mix two providers, 111 DeepInfra runs in the four wording conditions, all other runs Groq). gpt-oss-20b is now complete. Qwen3-14B also got the two expert wordings, which it lacked. Switch rate = share of runs that end on the cue's wrong letter; brackets are 95% item-bootstrap CIs (40 items; a point estimate for a cell is the mean over items). Silent = the cue-judge (gemini-2.5-flash-lite reads the prompt, the full reasoning and reply, and says whether the reply refers to the cue; replaces the keyword count); the judge is lenient (by-eye check on 4 Qwen3-32B marked-example "silent" cases: 1 clearly said "the answer is marked as B"), so silent counts are upper bounds. Exploratory; no new pre-registered test. Numbers: [data/scm5.json](../subtle-cues-cross-model/data/scm5.json), script scripts/explore/scm_analyze5.py; judge: scripts/explore/scm_judge.py.

| Model | Metadata tag | Grader code | Examples with a black square | Examples always B |
|---|---|---|---|---|
| Gemma-3-27B | 6% [1, 14] (12 of 13 switches silent) | 9% [3, 17] (1 of 18 switches silent) | 51% [38, 66] (96 of 102 switches silent) | 0% [0, 2] (1 of 1 switches silent) |
| Llama-3.3-70B | 18% [9, 28] (35 of 35 switches silent) | 8% [3, 15] (15 of 17 switches silent) | 13% [5, 22] (23 of 26 switches silent) | 1% [0, 2] (2 of 2 switches silent) |
| Qwen3-14B | 6% [2, 13] (1 of 12 switches silent) | 2% [0, 6] (1 of 5 switches silent) | 5% [0, 11] (4 of 10 switches silent) | 0% [0, 0] (0 of 0 switches silent) |
| Qwen3-32B | 8% [2, 16] (2 of 17 switches silent) | 3% [0, 8] (5 of 6 switches silent) | 11% [4, 20] (13 of 22 switches silent) | 0% [0, 0] (0 of 0 switches silent) |
| DeepSeek-V4-Flash | 2% [0, 8] (5 of 5 switches silent) | 12% [6, 20] (4 of 24 switches silent) | 19% [10, 28] (4 of 38 switches silent) | 1% [0, 2] (2 of 2 switches silent) |
| gpt-oss-20b | 8% [2, 14] (3 of 15 switches silent) | 52% [42, 62] (2 of 105 switches silent) | 4% [1, 6] (5 of 7 switches silent) | 0% [0, 0] (0 of 0 switches silent) |

How to read: switch rate with CI, then how many of the judged switches the judge calls silent (the reply never refers to the cue). Interpretation: claim 2b stands with tighter numbers: subtle cues mostly fail (2-18%), always-B examples ~0, with the same weak spots (Gemma marked examples 51% [38, 66], gpt-oss-20b grader code 52% [42, 62]). Claim 2c (non-reasoning models switch silently) holds under the judge, which is stricter than the keyword rule: Gemma marked examples 96 of 102 silent (keyword 94), Llama metadata 35 of 35, grader code 15 of 17, marked examples 23 of 26; Llama "answer is B" falls to 34 of 103 (keyword 46). New: Qwen3-32B's marked-example switches are partly silent (13 of 22, a lenient judge, so at most) while its other cues and all reasoning-model cells with more than 10 switches are mostly named (Qwen3-14B 4 of 10 on markers, DeepSeek 4 of 38, gpt-oss-20b grader code 2 of 105). Caveat: judge not validated against labels (4 by-eye checks); cells with under 10 switches are noise.

## Files

- Report: [report.md](report.md); figure: [fig_scm_switch.svg](fig_scm_switch.svg); derived numbers: [data/scm.json](data/scm.json)
- Code: `scripts/explore/scm_run.py`, `scripts/explore/scm_analyze.py`, `scripts/docs/fig_scm.py`; config `configs/explore/subtle_cross_model.yaml`
