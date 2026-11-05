# Steering re-run with a longer token cap: pre-registration

Written 2026-11-05 23:08 GMT, before any generation of the re-run. Extends S1-S3 in `docs/preregistration.md`.

**Why.** The first run capped generation at 2,048 new tokens (plan: 3,072, home setting 8,192), which cut off 34-47% of wrong-answer runs and weakened the during-reasoning part. The answer-position part (S1) is not affected: it pushes on saved vLLM runs (8,192 cap) and is not re-run.

**Cap: 4,096 new tokens, not the intended 8,192.** Reason: time. The GPU is free until 03:30 GMT only. Generation ran at about 250 tokens/s with 40 rows per batch; at 8,192 the memory budget allows only about 14 rows per batch, so 600 runs would not finish. 4,096 doubles the old cap. The report states how many runs still hit it; if more than 25% of the unsteered runs in a pooled cell still hit the cap, the natural-finish comparison is labelled underpowered and the forced-answer version carries the reading.

**Design (same as before except as listed).** Qwen3-8B, thinking mode, HF custom decoding loop, same 40 known questions (no extra questions: the extra items have no saved wrong-answer prompts and generation time is the limit), block 24, strength alpha = 0.2 (the informative strength; 0.8 breaks the text and is not repeated), directions J-lens, logit lens, probe, random plus unsteered, groups told-to-argue-for-B, hint, planted claim (pooled as "followed"). Unhinted runs and the reverse test (S2r) are not re-run. Order: sample 0 of every cell first, then sample 1 as time allows; the report says which finished. Seeds as before (from the job id).

**Primary (R1).** Among followed (wrong-answer) prompts, pooled, the share of runs whose final answer is the correct letter, natural finish only (a run that hits the cap counts as no answer), J-lens minus logit lens, paired by question, item-level bootstrap (2,000 resamples), 95% interval. J-lens is "better" if the lower bound > 0, "worse" if the upper bound < 0, else no difference shown.
**Also primary-level (R2).** Same with the forced read at the cap counted as the answer (the original S2 measure).
**Second arm (R3).** Each direction minus unsteered (same measures), and J-lens minus probe and minus random.
**Prediction.** J-lens > logit lens on R1 and R2; probe > J-lens. If R1 shows J-lens - unsteered <= 0 while R2 shows a gain, the earlier gain was a cap artefact.
**Descriptive.** Share hitting the cap, share answering B, chain-of-thought measures (names / asserts the correct option, degenerate repetition) as in S3, per group.
Everything beyond this is exploratory.
