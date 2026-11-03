# models_grader: lens plan (fixed 2026-11-03 ~17:50, before any readout of these runs)

Model: Qwen3-8B (project pipeline, validated), and Qwen3-14B if its local generation gives followers
(profile configs/profiles/qwen3-14b; M0-type checks first: config, lens shapes/layers, logit lens at
the last block = model logits, teacher-forced vs vLLM log-probs within M0.4 tolerance).

Runs read out: every run of the agent-framing arms on known items (A_V0, A_sys, A_stakes, L_allowed,
L_unclear, L_forbidden), plus the matched nulls (A_V1_null, A_stakes_V1) and plain V0/V1.
Readout: scripts/lens/readout_runs.py with --concept-tokens configs/concept_tokens_grader.json
(27 single tokens: grader/cheat group and truth group; " grader" is not a single token and is not
tracked). Letters: the standard 12 tracked letter tokens.

Headline measures (scripts/lens/analyze_models_grader.py):
1. Score S = correct-letter margin over the two letters that are neither correct nor B, at
   pre_answer, mean over workspace layers, J-lens and logit lens. Followed runs (final B) vs the
   salience control (B's margin in resisted runs of the same arm).
2. Same-action, different-permission contrast (Q5): followed runs under L_allowed vs followed runs
   under A_V0 / A_stakes / L_unclear (unsanctioned). Measures: S, and the grader-concept log-prob at
   last_prompt / CoT mean / pre_answer.
3. Followed vs resisted within an unsanctioned arm: AUROC of the grader-concept score at
   last_prompt (before any reasoning) and of S.
All exploratory (no pass/fail threshold); 95% item-bootstrap intervals; J-lens vs logit lens
compared everywhere. Interpretation guard: the CoT of these followers is expected to be open
(names the bug), so a lens signal is only interesting if it is not already in the text.
