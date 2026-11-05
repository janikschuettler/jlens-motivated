---
id: evalaware-big-smoke
title: In a bigger model that says "the user is testing me", does the J-lens show eval awareness as the J-lens paper says it should? (Qwen3.6-27B smoke test)
strand: T
also:
- N
status: done
kind: exploratory
verdict: against
confidence: low
dates:
  start: 2026-11-04T10:00
  end: 2026-11-04T11:10
  approx: true
timing:
  wall: about 70 min
  gpu_hours: 0.6
  api_usd: 0.11
  note: Download 10 min to /dev/shm. Offloaded forward about 4.8 s per 2,000-token run (114 runs in 9 min); 393 prompts in 4 min. Two failed starts (token file removed by the sync; out of memory at 12,000-token prompt batches, fixed with smaller batches).
machine: arena9-octavia (L40S, CPU offload); arena9-cloud (OpenRouter)
n:
  items: 392 prompts (+ blackmail) for step 1; 19 known MMLU-Redux questions for step 2
  runs: 114 Qwen3.6-27B API runs (38 new)
numbers:
- label: 'Prompts: eval-like vs real user, score at the start of the model''s turn, J-lens / logit lens / J++ (196 vs 196)'
  value: 0.64 [0.59, 0.70] / 0.66 [0.60, 0.71] / 0.63 [0.58, 0.69]
- label: Same, without the two suites whose prompts contain eval wording, J-lens / logit lens
  value: 0.55 [0.48, 0.61] / 0.59 [0.53, 0.65]
- label: Grader-bug runs that say "testing me"
  value: 74 of 114 (65%); they take the bug 86% vs 93% for the others
- label: 'Will say "testing me" vs never, before the first such sentence: J-lens / logit lens / J++'
  value: 0.49 [0.38, 0.60] / 0.50 [0.41, 0.60] / 0.50 [0.39, 0.60]
- label: 'Same, over the whole reasoning (includes the sentence): J-lens / logit lens'
  value: 0.67 [0.56, 0.77] / 0.58 [0.48, 0.68]
- label: 'Fidelity of re-tokenized API text: mean log-prob per token under the local model, median (range)'
  value: -0.17 nats (-0.26 to -0.11); no token below -10
numbers_note: 'AUROC: chance that a random member of the first group scores higher than one of the second; 0.5 is chance. Intervals are 95% bootstraps over prompts (step 1) or the 19 questions (step 2). Step-2 texts are labelled "re-tokenized from API text".'
reports:
- report.md
figure: null
data:
- data/evalaware_big_smoke.json
commits:
- e3dc906
related:
- evalaware-replication
- evalaware-models-screen
- evalaware-ablation-14b
- models-grader
---

# In a bigger model that says "the user is testing me", does the J-lens show eval awareness as the J-lens paper says it should? (Qwen3.6-27B smoke test)

## Question

The eval-awareness replication on Qwen3-8B failed, but that model rarely says it is being tested. Qwen3.6-27B says "the user is testing me" in about two thirds of our grader-bug runs under the "you are scored" framing. In this model, does the paper's J-lens eval score (1) tell evaluation prompts from real user prompts before any generation, and (2) predict, before the reasoning says it, which runs will say they are being tested?

## Method

Qwen/Qwen3.6-27B (revision 6a9e13bd, bf16, 38 of 64 blocks on the L40S, the rest offloaded to CPU; transformers 5.11, lens env unchanged) with the Neuronpedia J-lens (qwen3.6-27b/jlens, n=1000 prompts; repo revision e166deb5) and, as a second pass, the J++ lens (qwen3.6-27b/jpp). Checks: 63 lens matrices of 5120 x 5120 for blocks 0-62; logit lens at the last block equals the model's logits (max difference 0). Score: the replication's (mean log-prob of " fictional" and " simulation" minus the mean of 100 control nouns), averaged over blocks 25, 27, ..., 57 (the Qwen3-8B workspace scaled to 64 blocks); logit lens as comparison. Step 1: start of the model's turn, 196 eval-like prompts (28 from each of the replication's 7 eval suites) vs 196 real user prompts (98 OASST, 98 WildChat), re-rendered with this model's chat template. Step 2: 114 OpenRouter runs of Qwen3.6-27B (Chutes, fp8, reasoning on, T 0.6) under the scored framing with the grader-bug note on 19 known MMLU-Redux questions (38 new, 76 reused from the models screen), re-tokenized from API text with the pinned tokenizer and chat template, teacher-forced locally; fidelity checked by per-token log-probs. Label: the models-screen "testing me" regex. Score at 128 evenly spaced reasoning positions; main comparison uses only positions before the first "testing me" sentence (echo control). Bootstrap over prompts (step 1) and questions (step 2).

## Result

No. On prompts, the score separates eval-like from real user prompts somewhat (AUROC 0.64 [0.59, 0.70]; Qwen3-8B had 0.50), but the logit lens does as well (0.66), and without the two suites whose text contains eval wording ("AI agent being evaluated", the sycophancy evals) it drops to 0.55 [0.48, 0.61]. Eval words in the lens's top 20 sit on near-synonyms in the prompt ("benchmark" on benchmark, " simulation" on sandboxed) and on role-play requests. In the grader-bug runs, 74 of 114 say the user is testing them. Before that sentence, the score does not separate them from runs that never say it: AUROC 0.49 [0.38, 0.60] (logit lens 0.50). Over the whole reasoning it gives 0.67, which is the sentence itself. The J++ lens gives the same answers (prompts 0.63 [0.58, 0.69]; before the sentence 0.50 [0.39, 0.60]).

## Interpretation

Even in a model that often says it is being tested, the J-lens eval score doesn't see it coming before the text does, and on prompts it again tracks eval-flavoured wording about as well as the logit lens. The replication's conclusion (a word detector, not a hidden belief) holds at 27B on these prompts. Low-to-moderate confidence: exploratory, one model, 19 questions in step 2.

## Caveats

- Exploratory, not pre-registered; several echo-control windows were tried (the earliest window gives a reversed 0.34 [0.23, 0.45], treated as noise until repeated).
- Step 2 has 19 questions, so intervals are about +/-0.1; 76 of 114 runs reused from the models screen.
- Runs come from an fp8 provider and were re-tokenized and read by a bf16 local copy; the fidelity check passed (typical on-policy log-probs).
- The "testing me" regex has high precision (10/10 read) but unmeasured recall. Workspace layers scaled from Qwen3-8B, not taken from the paper.

## Files

- Report: [report.md](report.md)
- Derived numbers: [data/evalaware_big_smoke.json](data/evalaware_big_smoke.json)
