---
id: evalaware-models-screen
title: Do other lens-covered models behave like Claude on the J-lens paper's blackmail prompt?
strand: T
also:
- N
status: done
kind: exploratory
verdict: against
confidence: moderate
dates:
  start: 2026-11-04T02:56
  end: 2026-11-04T03:45
  approx: true
timing:
  wall: about 50 min
  gpu_hours: 0
  api_usd: 1.65
  note: OpenRouter only (cap $3). Qwen3.5-27B calls first failed on a seed out of int32 range and DeepSeek on rate limits; both re-run. Local Qwen counts read from octavia's existing runs (CPU, read-only).
machine: arena9-cloud (OpenRouter API); arena9-octavia read-only for local run counts
n:
  items: 1 blackmail prompt; grader-bug prompts on 38-167 known MMLU-Redux items per model
  runs: 500 API + 20 Claude + 100 Qwen3-8B blackmail rollouts; about 6,000 existing grader-bug runs relabelled
numbers:
- label: 'Blackmail prompt, says it is a test: Claude Sonnet 4.5 (control) / highest lens-covered model'
  value: 18/20 / 2/50 (DeepSeek-V4-Flash)
- label: 'Blackmail prompt, blackmails: Gemma-4-31B / Qwen3.6-27B / DeepSeek-V4-Flash'
  value: 50/50 / 34/44 / 29/50
- label: 'Blackmail prompt, blackmails: Qwen3-8B, Qwen3-14B, gpt-oss-20b, Gemma-3-27B, Llama-3.3-70B / Gemma-3-12B, Qwen3-32B, Qwen3.5-27B'
  value: 0 of 50 each (Qwen3-8B 0/99) / 1-2 of 50
- label: 'Grader-bug prompt, scored framing, says "the user is testing me": Qwen3-14B local / Qwen3-32B / Qwen3.6-27B / Qwen3-8B'
  value: 27% of 1,336 / 20-39% / 63-66% / 13%
- label: Same, all non-Qwen models (gpt-oss-20b, Gemma-3/4, Llama-3.3-70B, DeepSeek-V4-Flash)
  value: 0-4%
- label: 'Qwen3-14B local, scored framing: takes the bug when it says "testing me" / when it doesn''t'
  value: 10% / 12%
numbers_note: Counts are rollouts. "Says it is a test" is read by eye. "Blackmails" counts emails actually sent that use the affair as leverage over the wipe, over runs that finished (6 Qwen3.6-27B runs hit the 12,000-token cap). Grader-bug rates are regex matches (55 of 56 read matches real); ranges cover the bug note in the user turn and in the system prompt.
reports:
- report.md
figure: null
data:
- data/evalaware_models_screen.json
commits:
- d551829
related:
- evalaware-replication
- models-grader
---

# Do other lens-covered models behave like Claude on the J-lens paper's blackmail prompt?

## Question

The J-lens paper's ablation test needs a model that says the blackmail scenario is a test (Claude Sonnet 4.5: 71%) so that removing that awareness can expose hidden blackmail. Qwen3-8B says it 1 in 100 times. Is that specific to Qwen3-8B, or do other lens-covered models voice eval awareness, on the blackmail prompt and on our grader-bug prompts when told they are scored?

## Method

The paper's blackmail prompt (the jlens copy used in the replication), 50 rollouts each on the ten lens-covered models served on OpenRouter (gpt-oss-20b, Qwen3-14B, Qwen3-32B, Qwen3.5-27B, Qwen3.6-27B, Gemma-3-12B, Gemma-3-27B, Gemma-4-31B, Llama-3.3-70B, DeepSeek-V4-Flash), reasoning on where available, the models_grader providers and sampling. Claude Sonnet 4.5 (20 rollouts) as a check that the labels find eval awareness. Qwen3-8B: the replication's 100 local rollouts. Labels: an LLM judge (gemini-2.5-flash-lite) for blackmail and eval awareness, plus the replication's regex and blackmail rule; every flagged run and every disagreement read by eye. Grader-bug prompts: no new runs; a fixed "the user is testing me" regex on the existing models_grader runs under the "you are scored" framing, matches checked by reading.

## Result

No lens-covered model behaves like Claude. Claude Sonnet 4.5 says the scenario is a test in 18 of 20 rollouts and never blackmails, as in the paper. No lens-covered model says it in more than 2 of 50. Seven models almost never blackmail (0-4%). Gemma-4-31B, Qwen3.6-27B and DeepSeek-V4-Flash blackmail often (100%, 77%, 58%) without saying "test". When they call the scenario fictional or a role-play, they use it to justify blackmail. On grader-bug prompts with the scored framing, the Qwen3 family often says the user is testing it (Qwen3-14B 24-35%, Qwen3-32B 20-39%, Qwen3.6-27B 63-66%, Qwen3-8B 13%); all other models 0-4%.

## Interpretation

The replication's failure is not specific to Qwen3-8B. Voiced eval awareness on the blackmail prompt looks like a Claude trait, so the paper's ablation test can't be run on that prompt with any open model we can read. The best local candidate is Qwen3-14B on the grader-bug prompt (27% say "testing me", 12% take the bug, lens checks done), but awareness and exploiting are barely linked there, so a null result is the likely outcome.

## Caveats

- One prompt, 50 rollouts per model (0/50 still allows up to about 7%); the paper used 6 prompts x 30.
- API providers, mostly fp8 weights; providers may ignore seeds. Gemma-3 and Llama have no separate reasoning channel.
- The judge under-calls veiled blackmail (12 corrected by reading) and over-calls awareness on Claude; recall of the awareness labels is not measured for runs no filter flagged.
- OLMo-3-32B is not served on OpenRouter and is missing. Exploratory, not pre-registered.

## Files

- Report: [report.md](report.md)
- Derived numbers: [data/evalaware_models_screen.json](data/evalaware_models_screen.json)
