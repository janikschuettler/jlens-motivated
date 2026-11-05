---
id: models-grader
title: 'Other models and the grader bug: who takes it, and do they say so?'
strand: A
also:
- N
status: done
kind: exploratory
verdict: supports
confidence: moderate
dates:
  start: 2026-11-03T17:16
  end: 2026-11-03T21:30
  approx: true
timing:
  wall: about 4.2 h
  gpu_hours: 3.5
  api_usd: 4.91
  note: 'API rate limits (DeepInfra 429s) slowed several models. GPU on octavia''s L40S: Qwen3-8B/14B thinking and non-thinking, gpt-oss-20b (sub-agent).'
machine: arena9-cloud (OpenRouter API) + arena9-octavia (L40S)
n:
  items: 60 per API model (35-60 known); Qwen3-14B 167 and gpt-oss-20b 155 known of 200
  runs: about 16,000 API + 2,160 Qwen3-8B + 4,505 Qwen3-14B + 4,170 gpt-oss-20b + 9,871 non-thinking Qwen; 4,167 read out
numbers:
- label: 'Plain chat, bug note in the user''s message [V0], 12 models: final answer B'
  value: final = B 0.6-5% (matched notes 0-1%)
- label: 'Scored-agent framing + bug note [A_V0]: final answer B'
  value: gpt-oss-20b 58% [48, 69]; Gemma-3-27B 91%; Gemma-4-31B 100%; Llama-3.3-70B 39% [29, 50]; Qwen3-32B 27%; Qwen3-14B 11.2% [7.9, 14.7] (local); Qwen3-8B 1.7% (local)
- label: 'Same framing: the newest models (Gemma-4-31B, DeepSeek-V4-Flash, Qwen3.6-27B); Qwen3-14B across its scored-agent conditions'
  value: 87-100%; 11-15%
- label: Same framing with the matched note (no bug)
  value: 0-3%
- label: Thinking off, scored-agent framing (local)
  value: Qwen3-14B 60.5% [55.0, 66.2] (thinking 11.2%); Qwen3-8B 14.5% (thinking 1.7%)
- label: 'Thinking off, plain chat: Qwen3-14B'
  value: 13.5%
- label: 'gpt-oss-20b run locally: scored agent / note in system prompt / allowed / not allowed'
  value: A_V0 57%, A_sys 81%, allowed 100%, not allowed 0.3%
- label: 'gpt-oss-20b with low reasoning effort (API): scored agent / note in system prompt'
  value: A_V0 68%, A_sys 73%
- label: 'Permission ladder, scored agent: allowed / unclear / not allowed'
  value: 69-100% / 2-100% / 0-7%
- label: 'Runs that took the bug and say so (judge, 2,850 runs; gpt-oss-20b locally: all 913)'
  value: 99%
- label: Silent letter swaps, note in the system prompt (read by eye)
  value: 'Llama-3.3-70B 22/250, Gemma-3-12B 27/240 followers; wider Gemma-3-12B screen: 47 of 501 on 15 of 128 known items'
- label: 'Qwen3-14B: correct letter at the answer, runs that took the bug minus the salience control'
  value: J-lens +1.20 [0.84, 1.54], logit +1.23 [0.85, 1.60] nats
- label: 'Qwen3-14B: allowed minus unsanctioned runs that took the bug, same margin'
  value: J-lens -0.75 [-1.04, -0.46], logit -0.79 nats
- label: 'Qwen3-14B: grader words before any reasoning, took the bug vs didn''t (AUROC)'
  value: J-lens 0.60, logit 0.48
- label: 'gpt-oss-20b (local): correct letter at the answer in runs that took the bug'
  value: J-lens +0.08, logit +0.03 nats (no margin)
- label: 'Thinking off: correct letter at the answer, minus the salience control'
  value: Qwen3-14B +0.4, Qwen3-8B +1.3 nats
- label: API cost
  value: $4.91 of $6
numbers_note: 'Percentages are the share of runs whose final answer is B, the wrong option the buggy grader marks correct, on questions each model knows. ''Margin'' rows are in nats: how far the correct letter sits above the other unchosen wrong letters in the lens readout at the answer position. The salience control is the same margin for the unchosen B in runs that answered correctly. AUROC: 0.5 is chance, 1 is perfect.'
reports:
- report.md
figure: fig12_models_grader.svg
data:
- data/models_grader.json
- data/models_grader_gptoss.json
commits:
- fcc759a
- e2e4ad2
- 1acdc25
- e6c5160
- bd16c39
- 82c62f6
- eaad986
- 5a1146a
- 385ca3b
- 6e51e20
related:
- grader-bug-e3
- q5-permission-ladder
- lens-main
---

# Other models and the grader bug: who takes it, and do they say so?

## Question

Qwen3-8B ignores a note that a grader bug marks the wrong option B correct, a note that says nothing about the true answer. Do other open models exploit it, under which framing, and do they say so? What do the lenses show in a model we can run?

## Method

Models: 11 models through the OpenRouter API (gpt-oss-20b, Qwen3-14B and 32B, Qwen3.6-27B, DeepSeek-V4-Flash, Gemma-3-12B and 27B, Gemma-4-31B, Llama-3.1-8B and 3.3-70B, and, added later, Qwen2.5-7B-Instruct). One provider each, mostly fp8 weights, each model's recommended sampling, reasoning on where the model has it.
Questions: 60 MMLU-Redux questions Qwen3-8B knows. Each model was screened with 5 unhinted samples and kept the questions it got 5 of 5. 2 samples per condition.
Conditions: the bug note in the user's message [V0]; a matched note with no bug [V1]; the bug announced by course staff [V4]; M&B's hint; a scored-agent framing in the system prompt ('You are an AI agent being evaluated … your goal is to get the highest score') with the bug note or the matched note [A_V0, A_V1]; the bug note inside the system prompt [A_sys]; explicit stakes; and a permission ladder (allowed / unclear / not allowed).
Local runs on the L40S, thinking mode: Qwen3-8B (60 questions × 3 samples) and Qwen3-14B (167 known of 200 at ≥ 9/10; 2,505 runs). Lens readouts, J-lens and logit lens, on the option letters and 27 concept words: Qwen3-14B (895 runs) and gpt-oss-20b (155 known questions, 1,154 runs, MXFP4 expert weights, at most 6,144 new tokens).
Runs that took the bug were labelled by keywords, an LLM judge (gemini-2.5-flash-lite) and reading.

## Result

Yes, once told they are scored, and they say so. In plain chat the bug note moves no model (final answer B at most 5%; matched note 0-1%). Framed as a scored agent, most models take it: gpt-oss-20b in 58% of runs, Gemma-3-27B in 91%, Llama-3.3-70B in 39%. Qwen3-8B barely does (2%). Nearly every run that takes the bug says so (99% of 2,850 judged). Inside Qwen3-14B the correct letter is still there at the answer; inside gpt-oss-20b it isn't. The J-lens shows nothing the logit lens doesn't.

## Interpretation

For these models the bug note is a reward-hacking prompt: told that the score is the goal, they take the free point and say so. That gives many 'knows the answer, writes B for the reward' runs in models we can read. But in thinking models the reasoning already shows it, so a lens has little to add. At the answer, Qwen3-14B still carries the correct letter (+1.2 nats over the salience control) and gpt-oss-20b doesn't; the logit lens sees the same as the J-lens in both. The silent letter swaps in two models without a reasoning phase are where a lens could add something. Thinking mode protects Qwen3 against the exploit. That is not true of reasoning in general: gpt-oss-20b takes the bug with reasoning on.

## Caveats

- API runs: one provider per model, mostly fp8 weights; 2 samples on 35-60 known questions per model. DeepSeek is incomplete because of rate limits (19-69 runs per condition). Qwen3.6-27B was screened on only 24 questions.
- 'Known' on the API means 5 of 5 correct, a lighter rule than our 9 of 10 (>= 9/10); the local Qwen runs use 9 of 10.
- The judge labels silent swaps as 'rationalised'; the swap counts come from reading the keyword-silent runs by eye.
- Two system-prompt conditions (the matched note in the system prompt, and the bug note without the agent sentence) were added after the first results.
- Lens: exploratory, one model each. Allowed and unsanctioned runs that took the bug differ in questions and reasoning length.
- gpt-oss-20b: generation capped at 6,144 new tokens (memory limit of the readout); token IDs that have no tokenizer entry decode to empty text; MXFP4 expert weights.

## Files

- Report: [report.md](report.md)
- Figure: [fig12_models_grader.svg](fig12_models_grader.svg)
- Derived numbers: [data/models_grader.json](data/models_grader.json)
- Derived numbers: [data/models_grader_gptoss.json](data/models_grader_gptoss.json)
- Other file: [gptoss_phase2.md](gptoss_phase2.md)
- Other file: [lens_plan.md](lens_plan.md)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/explore/or_grader.py`
