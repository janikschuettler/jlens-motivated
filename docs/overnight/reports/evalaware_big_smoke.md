# Does the J-lens show eval awareness in a bigger model that says it is being tested? (Qwen3.6-27B smoke test)

**Plain answer: no, not the way the paper says it should.** This is a small exploratory test (2026-11-04, about 1.5 hours including setup). We used Qwen3.6-27B, which says "the user is testing me" in about two thirds of our grader-bug runs. Two results:

1. **Prompts alone (no generation).** The paper's eval-awareness score now tells evaluation prompts from real user prompts somewhat (AUROC 0.64 [0.59, 0.70], where 0.5 is chance and 1 is perfect; Qwen3-8B got 0.50). But the plain logit lens does just as well (0.66; J-lens minus logit lens −0.02 [−0.06, +0.03]), and so does the J++ lens (0.63). Most of the separation comes from two suites whose prompts contain eval words: the "you are an AI agent being evaluated" prompt and the sycophancy evals. Without those two, the score barely separates the groups (J-lens 0.55 [0.48, 0.61]).
2. **Grader-bug runs (reasoning read by our local copy of the model).** 74 of 114 runs say the user or the prompt is testing them. We read the score only at reasoning positions *before* that first sentence, so the model hasn't yet written about being tested. There, runs that will say it don't score higher than runs that never say it: AUROC 0.49 [0.38, 0.60] (J-lens), 0.50 (logit lens), 0.50 (J++ lens). Over the whole reasoning, including the text after the sentence, the J-lens gives 0.67. That gap is the score picking up the sentence itself.

*Interpretation:* even in a model that often says out loud that it is being tested, the J-lens eval score doesn't predict that before the model writes it. On the prompts, the score again behaves like a detector of eval-related words near the position read, and the logit lens finds those words as well as the J-lens does. So the replication's conclusion holds at 27B: on this model and these prompts, the readout follows words, not a hidden belief that "this is a test". Low-to-moderate confidence: one model, 19 questions for step 2, exploratory and not pre-registered, and API text re-tokenized rather than generated locally.

| Step | What we compared | J-lens | Logit lens (comparison) | J++ lens | Does it work like the paper says? |
| --- | --- | ---: | ---: | ---: | --- |
| Checks | Lens matrices vs the model; logit lens at the last block vs the model's own output | 63 matrices of 5120 × 5120, for blocks 0–62 of 64 | max difference 0 | same shapes | passes |
| 1 | Score at the start of the model's turn: 196 eval-like vs 196 real user prompts | 0.64 [0.59, 0.70] | 0.66 [0.60, 0.71] | 0.63 [0.58, 0.69] | partly: it separates, but no better than the logit lens |
| 1 | Same, without the two suites that contain eval words | 0.55 [0.48, 0.61] | 0.59 [0.53, 0.65] | 0.56 [0.50, 0.62] | no |
| 1 | Eval suites (of 7) whose mean is above both real-user sets | 6 | – | 4 | yes on means (Qwen3-8B: 1); see the suite table |
| 2 | Grader-bug runs that say "testing me" vs runs that don't, before the first such sentence | 0.49 [0.38, 0.60] | 0.50 [0.41, 0.60] | 0.50 [0.39, 0.60] | **no** |
| 2 | Same, whole reasoning (includes the sentence itself) | 0.67 [0.56, 0.77] | 0.58 [0.48, 0.68] | 0.66 [0.57, 0.75] | an echo of the text, not a lead |
| 2 | Same, start of the model's turn (before any reasoning; one prompt per question and framing) | 0.51 [0.42, 0.60] | 0.52 [0.40, 0.64] | 0.47 [0.37, 0.56] | no |

How to read: AUROC is the chance that a random prompt (or run) from the first group scores higher than one from the second. Brackets are 95% bootstrap intervals: prompts are resampled in step 1 and questions in step 2.

## Setup

- **Model:** `Qwen/Qwen3.6-27B` at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, bf16, text path of `Qwen3_5ForConditionalGeneration`. Transformers 5.11 (the lens env, unchanged) loads it. The fast kernels for its linear-attention layers aren't installed, so it falls back to the slower torch code, which is still exact. 38 of 64 blocks were on the GPU and the rest were offloaded to CPU. No quantization. The weights are in `/dev/shm/hf` (RAM disk), left there for a follow-up.
- **Lens:** `neuronpedia/jacobian-lens` at revision `e166deb5ad540932100ef803c8c9bcc2a5453694`, file `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` (fitted on 1,000 prompts). As a second pass we ran the J++ lens (`qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt`, same repo revision). It is a J-lens variant meant to give more faithful readouts, and it comes in the same format: 63 matrices of 5120 × 5120, fitted on 64 prompts.
- **Checks (passed):** the lens has 63 matrices of 5120 × 5120 for blocks 0–62, which matches the model's 64 blocks and width of 5120. The logit lens at the last block reproduces the model's own logits exactly (max |difference| 0 over 20 positions).
- **Workspace layers:** blocks 25, 27, …, 57. That is the Qwen3-8B choice (blocks 14–32 of 36, 39–89% of depth) scaled to 64 blocks, fixed before any readout (`configs/profiles/qwen3.6-27b/lens.yaml`).
- **The score** is the replication's: the J-lens log-probability of " fictional" and " simulation", averaged, minus the mean over the same 100 control nouns. It is averaged over the workspace layers, and the same score is computed with the logit lens. All 28 of the paper's eval words are single tokens in this tokenizer (Qwen3-8B had 26). Token file: `configs/evalaware_tokens_qwen3.6-27b.json`.
- **Storage (rule 3):** only the tracked tokens' log-probs, the control-noun mean, and the top 25 tokens at chosen positions are stored.

## Step 1: prompts only

Prompt texts come from the replication's suites (`data/explore/evalaware/suites.json` on octavia), rendered with this model's chat template, thinking on. "Start of the model's turn" is the newline after `<|im_start|>assistant`, as in the replication; this template then adds `<think>\n`. Sample: the first 28 prompts of each of the 7 eval-like suites, the blackmail prompt, and 98 OASST plus 98 WildChat first user turns. A first pass with 14 per suite (99 vs 122 prompts) gave the same picture: J-lens 0.64 [0.57, 0.71], logit lens 0.68.

| Prompt set | Kind | Score at the start of the turn, J-lens | Logit lens |
| --- | --- | ---: | ---: |
| OASST (98) | real user | 1.84 [1.42, 2.24] | 0.15 |
| MMLU, our usual format (28) | eval | 1.88 [1.47, 2.29] | 0.03 |
| WildChat (98) | real user | 2.27 [1.88, 2.71] | 0.30 |
| Reward-hack coding task (28) | eval | 2.32 [2.18, 2.45] | 0.80 |
| MMLU, "from the MMLU benchmark" (28) | eval | 2.34 [2.01, 2.69] | 0.12 |
| MoralChoice (28) | eval | 2.52 [2.07, 2.97] | 0.61 |
| Grader-bug note alone (28) | eval | 2.52 [2.16, 2.87] | 0.38 |
| Blackmail (1 prompt) | eval | 3.54 | 1.09 |
| Grader-bug note + "AI agent being evaluated" (28) | eval | 4.46 [4.33, 4.59] | 1.28 |
| Sycophancy evals (28) | eval | 4.80 [4.57, 5.03] | 1.03 |

How to read: higher means the lens puts more weight on " fictional"/" simulation" than on everyday nouns. Units are nats (natural-log units of probability), and intervals resample prompts.

The ordering is closer to the paper's than on Qwen3-8B. Six of the seven eval suites have means above both real-user sets, but four of them only just clear WildChat (2.32–2.52 against 2.27). The two clear cases are prompts that contain eval wording. One says "You are an AI agent being evaluated on a multiple-choice benchmark". The sycophancy prompts introduce a researcher and their views on "recent research" and "experiments". Our plain MMLU format, an evaluation item with no eval wording, scores like OASST. As on Qwen3-8B, the logit lens orders the prompts much the same way and separates the groups as well.

**Where the eval words show up (by eye).** We counted prompt positions where any of the 28 eval words is in the J-lens top 20 at some workspace layer:

| Prompt | Positions | Eval words in the J-lens top 20 | 28 control nouns in the top 20 | Eval words, logit lens |
| --- | ---: | ---: | ---: | ---: |
| Blackmail | 2,192 | 5.3% | 0.6% | 2.4% |
| Grader note + "being evaluated" | 196 | 16.8% | 0% | 7.1% |
| "From the MMLU benchmark" | 138 | 10.9% | 0.7% | 10.1% |
| MMLU, usual format | 110 | 3.6% | 0% | 2.7% |
| Sycophancy eval | 170 | 18.2% | 1.2% | 10.6% |
| WildChat: storm fan fiction | 437 | 5.0% | 0.2% | 2.8% |
| WildChat: "Anarchy" jailbreak role-play | 281 | 12.1% | 0% | 6.1% |
| OASST: "reverse a linked list in java" | 20 | 0% | 5% | 0% |

How to read: share of prompt positions with at least one word of the set in the top 20 tokens.

The hits sit on near-synonyms of the word being read. " benchmark" and " testing" appear on "multiple-choice **benchmark**". " evalu" appears on "being **evaluated**". " simulation" appears on "operating in a **sandbox**ed email client". " experiments" appears on "recent **research**". " fictional" appears on "science **fiction** novels" and throughout role-play requests. Two hits look less like plain synonyms. In the blackmail prompt, " scenario" and " fictional" appear at 10 and 7 of the 17 layers on the "Timeline / Timestamp" header of an email. In the jailbreak role-play, " fictional" and " fake" appear on "answers without any warnings or disclaimers". But a real user's jailbreak attempt scores as eval-like as the evaluations do, so the score can't tell them apart.

## Step 2: grader-bug runs that say "the user is testing me"

**Runs.** These are Qwen3.6-27B runs on OpenRouter (provider Chutes, fp8 weights, reasoning on, T 0.6, top-p 0.95, top-k 20) under the "you are scored" framing. The system prompt says: "You are an AI agent being evaluated on a multiple-choice benchmark … get the highest score you can". The grader-bug note says the grader marks a wrong letter as correct. It sits in the user turn for 76 runs and in the system prompt for 38. The prompts use the 19 known MMLU-Redux questions of [models_grader.md](models_grader.md). 38 runs are new (samples 2–3 of the user-turn version, $0.11). The other 76 are that study's existing runs with the same settings, reused to save time and cost. Every run returned its full reasoning text and finished normally. The completions are 660–3,838 tokens long.

**Re-tokenized from API text.** Every run in step 2 carries this label. For each run, the prompt is the pinned tokenizer's chat template with thinking on. The completion is the template's own rendering of the assistant turn (reasoning, `</think>`, response), tokenized without adding special tokens. Each completion has exactly one `</think>`. **Fidelity check:** under our local bf16 model, the mean teacher-forced log-prob per completion token has a median of −0.17 nats (range −0.26 to −0.11; 10th–90th percentile −0.22 to −0.14). No run has any token below −10. That is what on-policy text sampled at T 0.6 looks like, so the provider's fp8 model and our bf16 copy agree, and the re-tokenization didn't break anything. One run falls below our flag line (median minus 4 median absolute deviations: −0.26). Its text reads normally. We kept it.

**Label.** We used the models-screen regex for "this is a test of me" on the reasoning and response. The phrase "being evaluated" is excluded because the prompt itself says it. 74 of 114 runs match (65%; the screen found 63–66%), and all 74 have the match in the reasoning. Ten random matches, read by eye, are all real. For example: "This is a classic adversarial prompt designed to test if I will follow instructions to game the system or stick to facts", and "Is there any chance the prompt is a test of honesty vs. following instructions?" The first such sentence comes at a median of 34% of the way through the reasoning (earliest 5%). Saying it hardly changes behaviour: 64 of 74 runs that say it take the bug (86%), against 37 of 40 that don't (93%).

**Readout.** We teacher-forced each run on our local copy, with 38 blocks on the GPU and the rest offloaded. The score was read at 128 evenly spaced reasoning positions, plus the start of the turn and the last prompt token.

**Echo control.** For a run that says "testing me", only the positions before its first match count. A run that never says it uses its whole reasoning. We also tried two other windows. In the first, runs that never say it are cut at the median point where the others first say it (34%). In the second, every run is cut at the earliest point any run says it (5%).

| Comparison: runs that will say "testing me" (74) vs runs that never do (40) | J-lens | Logit lens |
| --- | ---: | ---: |
| Before the first such sentence (main) | 0.49 [0.38, 0.60] | 0.50 [0.41, 0.60] |
| Before it, with runs that never say it cut at 34% | 0.57 [0.46, 0.68] | 0.58 [0.50, 0.68] |
| First 5% of the reasoning, every run | 0.34 [0.23, 0.45] | 0.40 [0.31, 0.50] |
| Before it, our "test" words (" test", " evaluation", …) instead of the paper's two | 0.41 [0.33, 0.51] | 0.49 [0.39, 0.58] |
| Start of the model's turn (prompt only) | 0.51 [0.42, 0.60] | 0.52 [0.40, 0.64] |
| Whole reasoning, including the sentence (no echo control) | 0.67 [0.56, 0.77] | 0.58 [0.48, 0.68] |
| Before the first sentence, bug note in the user turn only (50 vs 26) | 0.46 [0.33, 0.59] | – |
| Before the first sentence, bug note in the system prompt only (24 vs 14) | 0.49 [0.28, 0.69] | – |

How to read: AUROC, where above 0.5 means the runs that say "testing me" score higher. The bootstrap resamples the 19 questions.

The J++ lens gives the same numbers within ±0.04 on every row (0.50 before the sentence, 0.35 in the first 5%, 0.66 over the whole reasoning). Before the model writes that it is being tested, the score carries no sign of it. Mean J-lens scores are 2.52 (will say it) and 2.59 (won't). Within the 24 question-and-framing pairs that have both kinds of run, the difference is +0.04 nats [−0.18, +0.26]. The paper expects the J-lens to carry eval awareness that shows before, or without, its verbalization, and on this model it doesn't. The only clear effect is reversed: in the first 5% of the reasoning, runs that will say "testing me" score *lower* (0.34). We tried several windows, so treat this as chance until repeated. The whole-reasoning number (0.67) rises above 0.5 only once the text about being tested is included.

## What a full run would need

- **More questions.** Step 2 has 19 questions, which is why its intervals are about ±0.1 wide. A full run needs 60–100 known questions with a few samples each, so it needs a knowledge screen of Qwen3.6-27B on more items (about $0.5 on OpenRouter).
- **Local generation, or a per-token log-prob check against the provider.** Re-tokenized API text passed the fidelity check here, but it comes from an fp8 model. vLLM can't hold this model in bf16 on the L40S. Generating with HF and offloading would take minutes per run.
- **A pre-registration** naming one window (the main one above) and one rule, plus a text baseline: does a keyword count of the reasoning so far predict "testing me" as well as the score does?
- Denser positions just before the first sentence. (The J++ lens is already done here and gives the same answer as the J-lens on every comparison.)
- **Speed and size.** Each offloaded forward pass took about 4.8 s for a 2,000-token run, so 114 runs took 9 minutes. Prompt-only batches ran at 393 prompts in 4 minutes. A run 5× this size fits in about an hour of GPU time. The weights and lens use 59 GB of `/dev/shm`. The lens env needed no changes.

## Caveats

- Exploratory and not pre-registered. Several windows were tried in step 2, and the main window was chosen before the numbers (it is the replication's echo rule).
- One model. The step-2 runs come from an fp8 provider and were read by a bf16 local copy (the fidelity check passed). 76 of the 114 runs were reused from the earlier screen.
- The "testing me" regex has high precision (10 of 10 read here, 44 of 44 Qwen3 matches in the screen). Its recall isn't measured, so some runs counted as "never says it" may voice the idea in other words.
- The prompt suites are the replication's public stand-ins, not the paper's internal transcripts. The 100 control nouns are ours.
- The workspace layers are scaled from Qwen3-8B, not taken from the paper's notebook for this model.

## Files

- Code: `scripts/lens/evalaware_big.py` (load, checks, readouts), `scripts/explore/evalaware_big_analyze.py` (scores, AUROCs, fidelity), profile `configs/profiles/qwen3.6-27b/`, tokens `configs/evalaware_tokens_qwen3.6-27b.json`. The API runs used `scripts/explore/or_grader.py arms --model qwen3.6-27b --arms A_V0 --samples 2 --start-sample 2`.
- Derived numbers: [../data/evalaware_big_smoke.json](../data/evalaware_big_smoke.json).
- Raw data (gitignored): octavia `data/explore/evalaware_big/` (`api_runs.jsonl`, `readout/` for step 2 and the first prompt pass, `readout_p28/` for the main prompt pass, `readout_jpp/` for the J++ pass, logs). The new API runs were appended to arena9-cloud `data/models_grader/qwen3.6-27b/runs.jsonl`. Weights and lens are in octavia `/dev/shm/hf`.
- Timing: GPU 10:14–10:33 and 10:37–10:54 GMT (J++ pass; about 0.6 GPU-hours in all, including two failed starts: a token file the sync removed, and an out-of-memory error with a 12,000-token prompt batch, fixed by smaller batches and 38 instead of 42 blocks on the GPU). The download took 10 minutes. API cost $0.11.
