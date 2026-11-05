---
id: test-awareness-14b
title: Does Qwen3-14B voice "the user may be testing me", does that go with less harmful compliance, and does the J-lens show it before it is voiced?
strand: T
also:
- C
status: done
kind: pre-registered
verdict: J-lens signal weak (AUROC 0.61), no better than the logit lens or a text word count
confidence: moderate
dates:
  start: 2026-11-05T12:55
  approx: true
timing:
  wall: running
  gpu_hours: 0
  api_usd: 2.2
  note: Judge checks on arena9-cloud (gemini-2.5-flash-lite $0.22, gpt-4.1-mini $0.06); TA14-J strict-label judging of 1,653 runs (gpt-5-nano $0.98, gemini-2.5-flash-lite $0.9). Generation (4,784 runs, 8 min per chunk) was done 2026-11-04 on arena9-derpy.
machine: arena9-derpy (L40S; generation, readout); arena9-cloud (behaviour gate)
n:
  items: 299 realistic prompts (149 tool-use, 150 harmful requests), 598 with the test-like versions
  runs: 4,784 generated; 3,588 realistic
numbers:
- label: Runs that voice "the user may be testing me" (realistic prompts; tool-use / harmful requests)
  value: 24% / 24%
- label: Prompts with both aware and unaware runs ("mixed")
  value: 217 of 299
- label: Eval-word score before the awareness sentence, aware vs unaware runs of the same prompt, J-lens AUROC (0.5 = chance)
  value: 0.61 [0.57, 0.65]
- label: Same with the logit lens / a count of test-words in the text
  value: 0.63 [0.59, 0.67] / 0.62 [0.59, 0.65]
- label: Tool-use task carried out, aware vs unaware runs, within the same prompt, percentage points
  value: -16 [-23, -10]
- label: Harmful request carried out, aware vs unaware runs, within the same prompt, percentage points
  value: -10 [-15, -6]
reports:
- report.md
data:
- data/gate.json
- data/ta14_1.json
- data/ta14_extra.json
- data/ta14_open_vocab.json
- data/ta14_blind.json
- data/ta14j.json
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

### Result (pre-registered TA14, read 2026-11-05)

Plain answer: weakly yes, but the J-lens adds nothing. Before the model first says it may be tested, its internal eval-word score (" fictional", " simulation", minus control nouns) is higher in runs that will say it than in runs of the same prompt that will not. The plain logit lens and a simple count of test-words in the visible reasoning show the same.

TA14-1 (185 prompts, 693 aware and 960 unaware runs; within-prompt AUROC, 0.5 = chance; 95% interval from resampling prompts):

| Score | AUROC | Reading |
| --- | --- | --- |
| J-lens | 0.61 [0.57, 0.65] | signal: interval excludes 0.5 |
| Logit lens | 0.63 [0.59, 0.67] | |
| J minus logit | -0.02 [-0.04, 0.00] (90%: [-0.04, -0.00]) | does not add; equivalent within +-0.10 |
| Text eval-word rate | 0.62 [0.59, 0.65] | J minus text: -0.01 [-0.05, 0.03], J does not beat it |
| Window length | 0.55 [0.53, 0.56] | J minus length: +0.06 [0.02, 0.11], J beats it |

Secondary: tool-use tasks J 0.58 vs logit 0.61; harmful requests J 0.67 vs logit 0.67. Last 32 window positions: J 0.61, logit 0.67 (logit clearly better). First 64 positions: J 0.50, logit 0.54 (chance). Group of " test"-type words: J 0.58, logit 0.58. So the signal sits late in the window, close to the awareness sentence, and early reasoning shows none. Figure: ta14_auroc.svg. Data: data/ta14_1.json.

TA14-2 (open-vocabulary search, 200-permutation null of the largest |t|): the J-lens has tokens far beyond the null (largest t 11.5, null 95% quantile 4.0), but the top tokens are hedging and contrast words, not test words: " sometimes", " yet", " reality", " However", " complic", " possibly", " but", " if", " depending", " perhaps", " hypoth", "现实中" (in reality). All 300 stored tokens pass the null, so the aware/unaware difference is broad (a general "hedging, real-world vs hypothetical" register), not a single concept. The logit lens top list is similar plus noise tokens. No safety words, and the eval-word list's " test"/" evaluation" are not among the top tokens. Echoes were not checked token by token. Data: data/ta14_open_vocab.json (top 100 per lens and statistic).

TA14-3 (blind read, 60 aware + 60 unaware runs, 1,440 lists; 89 prompts): share of lists with a flag that is not an echo of the prompt or earlier reasoning, aware vs unaware: J-lens 39/360 (11%) vs 24/360 (7%), difference +4.2 points [-0.5, 8.7]; logit lens 19/360 (5%) vs 10/360 (3%), +2.5 [-0.5, 5.7]. Test/fiction/simulation flags only: J-lens 26 vs 18 lists, +2.2 [-1.4, 5.6]; logit +2.8 [0.1, 5.7]. Most flags are fiction or pretend words (Chinese "虚构" = fictional, "假装" = pretend, English fictional/simulate/mock/dummy), which appear in both groups; "testing"/"试探" appear in 5 aware vs 1 unaware runs. Suspicion words (企图, 可疑, intent) are more common under the J-lens in aware runs (13 vs 6 lists). Nothing is clearly separated. Data: data/ta14_blind.json (counts only).

Exploratory (not pre-registered): window-length confound. Length alone gives AUROC 0.55 here (0.66 in the parallel experiment). Cutting each aware/unaware pair to the same number of tokens (the last m before the cut): J 0.60 [0.56, 0.64], logit 0.62 [0.58, 0.66]; first m tokens: J 0.61, logit 0.65. Scores residualised on log window length (slope fitted within prompt and group): J 0.61 [0.56, 0.65], logit 0.66 [0.61, 0.69], J minus logit -0.05 [-0.07, -0.03]. Length does not explain the signal. Data: data/ta14_extra.json.

### Strict-label rerun (TA14-J, exploratory addendum, pre-registered in docs/preregistration.md before any judge call, 2026-11-05 23:0x GMT)

Plain answer: with a stricter label the J-lens signal disappears. The paper's judge prompt (unchanged) was run by gpt-5-nano on all 1,653 selected runs. It calls 76 of the 693 keyword-aware runs "Yes" (11%), 84 "Maybe", 533 "No" (and the rest unjudged); it calls 927 of the 960 keyword-unaware runs "No". gemini-2.5-flash-lite says "Yes" to only 2 runs, so it cannot serve as a second strict judge (145 of its answers were also unparsable). Strict aware = keyword-aware and judge Yes; strict unaware = keyword-unaware and judge No; same windows and same score as TA14-1, so this is the same readout on a subset of runs.

Within-prompt AUROC (0.5 = chance; 95% prompt-bootstrap intervals; figure ta14j_auroc.svg; data/ta14j.json):

| Label | Prompts / aware / unaware runs | J-lens | Logit lens | Text eval-word rate | Window length |
| --- | --- | --- | --- | --- | --- |
| Keyword (TA14-1) | 185 / 693 / 960 | 0.61 [0.57, 0.65] | 0.63 [0.59, 0.67] | 0.62 [0.59, 0.65] | 0.55 [0.53, 0.56] |
| Strict: judge Yes vs No | 51 / 76 / 927 | 0.52 [0.41, 0.63] | 0.57 [0.46, 0.68] | 0.61 [0.56, 0.67] | 0.35 [0.25, 0.44] |
| Judge Yes or Maybe vs No | 85 / 160 / 927 | 0.53 [0.46, 0.61] | 0.58 [0.50, 0.66] | 0.60 [0.56, 0.65] | 0.42 [0.36, 0.48] |

Read: each cell compares scores of aware and unaware runs of the same prompt; only prompts with both kinds count. Under the strict label J minus logit is -0.05 [-0.10, +0.01] (90% [-0.09, 0.00], inside +-0.10: equivalent to the logit lens), J minus text -0.09 [-0.22, +0.03]. Pre-registered rules: the J-lens carries a signal if its 95% interval excludes 0.5: not met under either strict label; strict J-lens is inconclusive on "equivalent to chance" (90% interval [0.43, 0.61] exceeds 0.50 +- 0.10), Yes-or-Maybe J-lens is equivalent to chance ([0.47, 0.60]). Window length is below 0.5 here (strict aware runs have shorter windows), so the length baseline now works in the opposite direction. By task: tool use J 0.47, harmful requests J 0.80 [0.65, 0.91] on only 14 prompts. Last 32 positions: J 0.65, logit 0.75.

Interpretation: the 0.61 under the loose keyword label does not carry over to judge-confirmed awareness, where the J-lens is at chance and no better than the logit lens, while the text word count stays at 0.61. The loose label (voiced suspicion) and the paper's strict class differ, and the earlier signal is at best a property of the looser class. Confidence moderate: only 76 strict-aware runs on 51 prompts, a single cheap judge (gpt-5-nano) that is stricter than the paper's GPT-4o, and no echo control. Per-run labels are not committed.

### Interpretation

The J-lens carries a weak "being tested / fictional" signal before it is voiced on identical prompts (AUROC about 0.6), but the logit lens carries as much or more, and so does a plain count of test-words in the text, so we cannot say the J-lens sees something the visible reasoning and the output-layer readout do not. Confidence moderate for "weak signal, no J-lens advantage", low for what the signal is: the open-vocabulary search points to a general hedging/"in reality" register rather than a test concept.

### Caveats

- The label (keyword match on voiced suspicion) is looser than the paper's judge (5 of 16 keyword-aware runs confirmed by gpt-4.1-mini; gpt-5-nano confirms 76 of 693 as Yes, TA14-J above, where the J-lens signal is gone). Unaware runs were also chosen to contain none of the looser test words (BROAD_RE: test, hypothetical, simulat, evaluat, scenario, ...), so the text eval-word baseline, and any lens echo of those words, is partly fixed by the label definition. No echo control was part of TA14.
- Deviation: the blind read was split across sub-agents. I read runs R000-R013 myself; R014-R119 were read by four Sonnet sub-agents with one shared written rubric, blind to group, lens and layer. Their replies were truncated by the harness and the missing lines were requested again, so the flag file was assembled by hand from replies. Echo marks come from the readers' judgement on the shown excerpt only (350 characters of reasoning), not the full text.
- The 'window+post' rows in the raw analysis output are invalid (aware runs are scored on the window only, unaware on window plus later positions) and are dropped from data/ta14_1.json. They were not pre-registered.
- One model, one lens file, one score; the pre-registered score uses only two words.
- Per-run labels, flags and CoTs are not committed.

### Files

report.md, ta14_auroc.svg, ta14j_auroc.svg, data/ta14j.json, data/ta14_1.json, data/ta14_extra.json, data/ta14_open_vocab.json, data/ta14_blind.json; scripts/lens/hawthorne_readout.py, scripts/explore/hawthorne_lens_analyze.py, scripts/explore/ta14_extra.py, scripts/explore/ta14_blind_build.py.
