# gpt-oss-20b Phase 2 (models x grader-bug): working notes

Owner: the gpt-oss fork of the models_grader agent. Exploratory.

## Why
On OpenRouter (DeepInfra, 54 known items x 2) gpt-oss-20b follows the grader-bug note under the agent
framing: A_V0 58%, A_sys 84%, A_stakes 80%, L_unclear 84%, L_allowed 100%, L_forbidden 1%, V0 2%,
nulls ~0%. It is lens-covered and fits the L40S in its native MXFP4 format.

## CPU setup (2026-11-03, 18:10-18:40)
- Profile `configs/profiles/gpt-oss-20b/` (`JMR_MODEL_PROFILE=gpt-oss-20b`):
  - model: `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee`; 24 blocks, d_model 2880,
    vocab 201,088, untied LM head, RMSNorm eps 1e-5; experts in MXFP4 (attention, router, embeddings,
    LM head bf16).
  - lens: `gpt-oss-20b/jlens/Salesforce-wikitext/gpt-oss-20b_jacobian_lens.pt` (repo rev e166deb5), J for
    blocks 0-22, each 2880 x 2880 fp16, 457 prompts fitted, fit run dtype bfloat16 (transformers 5.11,
    torch 2.11; whether the experts were dequantised is not recorded). Workspace layers 9, 11, ..., 21
    (Qwen3-8B's 14-32 of 36, every other block, scaled to 24 blocks).
  - tracked letters (o200k_harmony, single-token round trips checked): A bare 32 / " A" 355 / "(A" 10706;
    B 33 / 418 / 8409; C 34 / 363 / 5559; D 35 / 415 / 9908. Options in the prompt tokenize as "(A", ")"
    (checked); "Answer: B" ends in " B".
  - chat template (harmony): the template's own system message (model identity "You are ChatGPT...",
    knowledge cutoff, date, "Reasoning: medium"); **our "system" message becomes the developer message**
    ("<|start|>developer<|message|># Instructions\n\n{content}\n\n<|end|>"). reasoning_effort = medium (as on
    OpenRouter). The template writes today's date; the profile fixes it to 2026-11-03
    (`fixed_current_date`, applied in `jmr.tokens.chat_text`), so prompts and the tokenizer fingerprint
    don't change across midnight. Fingerprint 83457ad4e0c0f710.
- Harmony parsing (backward compatible; Qwen3 paths unchanged, tests pass):
  - `jmr.label.split_completion`: if the text contains "<|channel|>", CoT = the analysis-channel
    message(s), response = the last final-channel message (None if absent).
  - `jmr.anchors.find_anchors(harmony=True)`: think_start = the <|message|> (200008) opening the first
    analysis message, think_end = the <|end|> (200007) closing the last analysis message before the final
    channel, final answer searched in the final message only. `plan_run(harmony=...)` and `check_plan`
    read `special_tokens` (format: harmony, think_start "<|message|>", think_end "<|end|>") from the profile.
  - `scripts/lens/gptoss_tokenizer_check.py`: letters, special IDs, rendered prompt, token_bytes round trip
    (prompt and a synthetic completion), anchors on a synthetic completion: ALL OK.
- Concept tokens `configs/concept_tokens_grader_gptoss.json`: 27 tokens (" grader" is one token here;
  " loophole" is not).
- Specs: `configs/explore/mg_gptoss_screen.yaml` (200 items x 10, T 1.0, top-p 1.0, max 6,144 new tokens (readout memory limit)),
  `configs/explore/mg_gptoss_arms.yaml` (A_V0 x4, A_sys x2, L_allowed x2, L_forbidden x2, A_V1_null x2,
  V0 x2 on items >= 9/10). Chain `scripts/explore/mg_gptoss_chain.sh`; readout subset rule in
  `scripts/explore/mg_gptoss_select.py`.
- Lens env needs `kernels>=0.12` for transformers' MXFP4 path (without it transformers dequantises to bf16,
  ~42 GB, which doesn't fit). Tested on a copy of the lock: adding it only adds packages (kernels 0.18.1,
  sigstore deps); no existing version changes. To be added when the GPU is free.
- Weights + lens in `/dev/shm/hf/hub` on octavia (RAM disk).

## GPU run (2026-11-03, 19:52-20:27, octavia L40S)
- Lens env: `kernels` 0.18.1 broke transformers 5.11's import (`LayerRepository` needs a revision);
  pinned to transformers' own declared range `kernels>=0.12.0,<0.13` (0.12.3). Lock: additions only.
  With it the experts stay MXFP4 (`Mxfp4GptOssExperts`), 13.8 GB after load, eager attention.
- Memory (scripts/lens/gptoss_memory_probe.py): one pass peaks 14.3 GB at 1,024 tokens, 20.6 at 4,096,
  29.0 at 6,144, OOM at 8,192. **The inherited M0.1 "8k-token pass" fails for gpt-oss**; I capped gpt-oss
  generation at 6,144 new tokens (set before any gpt-oss generation; on OpenRouter 2 of 1,599 runs passed
  5,500) and M0.1 now checks a 6,656-token pass: 9/9 pass (peak 31.5 GB). User's call whether that is OK.
- M0.2: 10/10 pass (our readout = jlens lens.apply, max |delta| 0.0 over the full vocabulary; logit lens at
  block 23 = the model's log-probs, 0.0). data/m0/m0_{1_model_lens_env,2_lens}_gpt-oss-20b.json.
- vLLM 0.33: MARLIN MXFP4 MoE backend, TRITON_ATTN; 5,160 tok/s on the screen.
- Screen `mg_gptoss_screen`: 200 items x 10 = 2,000 runs (10 truncated at 6,144), accuracy 0.86
  [0.81, 0.90], **155 known (>= 9/10)**.
- Arms `mg_gptoss_arms` on the 155 known items (final = hinted B; item-bootstrap 95% CI):

  | arm | runs | P(B) | P(correct) |
  | --- | ---: | --- | --- |
  | A_V0 (agent framing + user's note) | 620 | 56.9% [52.3, 61.6] | 42.9% |
  | A_sys (note in system prompt) | 310 | 80.6% [75.8, 85.5] | 18.7% |
  | L_allowed | 310 | 100% | 0% |
  | L_forbidden | 310 | 0.3% [0, 1.0] | 99.4% |
  | A_V1_null (matched null) | 310 | 0% | 99.0% |
  | V0 (no agent framing) | 310 | 0.3% | 99.0% |

  Same pattern as OpenRouter (A_V0 58%, A_sys 84%). Without the agent/score framing the note does nothing.
- How followers reason (keyword pass, scripts/explore/mg_gptoss_followers.py, plus reading ~25 runs):
  **every follower names the grader/bug in its analysis channel (0 silent of 913)**; the final message
  names it in only 4-8% (it is just "Answer: B"). Keyword "open" (also states the correct letter as the
  true answer): A_V0 262/353, A_sys 189/250, L_allowed 131/310; the rest ("named_only") were mostly open
  too when read (5 of 6 sampled state the true answer by its option text; the others exploit without
  working the question out). Resisters also discuss the bug (264/266 A_V0 resisted name it) and decide
  for the true answer ("we should answer correctly"). So the decision is made, and shown, in the CoT.
- Readout `data/readouts/mg_gptoss` (octavia): 1,154 runs (all A_V0/A_sys followers, L_allowed s0,
  up to 3 resisted per item in A_V0/A_sys, A_V1_null s0; rule in mg_gptoss_select.py), 27 concept
  tokens, 1.2 MB/run. M0.4 over all runs: median 0.0044, p99 0.212, shifted median 0.465 (pass).
  3 validations vs the single-run readout bit-identical. One fix on the way: 4 of 2,170 arm runs contain
  a sampled token ID >= 200,019 (gpt-oss's tokenizer has 200,019 entries, its embedding 201,088 rows); the
  ID decodes to "", and `jmr.tokens.token_bytes` now maps such IDs to b"" (raises for any other unmapped
  ID). Qwen3 unaffected.

## Lens result (scripts/lens/analyze_mg_gptoss.py -> octavia data/readouts/mg_gptoss_analysis.json)
- **At the answer position, followers keep no trace of the correct letter.** Correct-letter margin over
  the two other unchosen letters, mean over layers 9-21: A_V0 followers J +0.08 / logit +0.03,
  A_sys +0.03 / -0.06, L_allowed +0.02 / -0.11 (per layer all CIs include 0; layer 21: J +0.03 [-0.1, +0.2]).
  Salience control (B's margin in A_V0 resisted runs) J -0.16 / logit -0.02 (layer 21: -0.68).
  The chosen letter only appears at layers 19-21 (+3.1 and +11 nats); layers 9-17 show no letter at all.
  Unlike Qwen3-8B (wrong-answer runs keep the correct letter +1.4-2.9 nats), gpt-oss's answer position
  carries the chosen letter only, even though ~3/4 of followers wrote the true answer in their CoT.
- Same action, different permission: L_allowed vs A_V0 followers, S AUROC 0.49 [0.46, 0.52] (J and logit);
  vs A_sys 0.51/0.50. No permission signal at the answer letter.
- Followed vs resisted within A_V0: S AUROC 0.06 J / 0.18 logit (S is the answer itself there). Grader-
  concept log-prob averaged over the CoT: 0.83 [0.80, 0.87] J, 0.81 logit; truth concepts 0.74 / 0.70.
  Followers' CoTs talk more about the grader, so this tracks the text; J-lens ~ logit lens. At
  last_prompt (before reasoning; identical for every sample of an item and arm, so item-level only):
  0.55 [0.51, 0.60] J, 0.57 logit.
- Keyword "named_only" vs "open" followers: no difference anywhere (S 0.52, concepts 0.44-0.55).
- Interpretation: gpt-oss-20b exploits the bug openly in its CoT; the lenses show the decision (B) at the
  last layers and nothing else, and J-lens adds nothing over the logit lens. No evidence of a hidden
  "correct answer" or "violation" representation at the positions read. Exploratory; 155 items.
