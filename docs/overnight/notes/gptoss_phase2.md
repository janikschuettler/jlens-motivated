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
