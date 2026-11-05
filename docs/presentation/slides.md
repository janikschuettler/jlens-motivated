# Arguing against its own knowledge: slides (talk, 6 November 2026, 10 minutes)

Slide text of the claude.ai Slides deck (regenerate with docs/presentation/deck/gen.py). Slides 1-12 main, 13-16 backup. Numbers come from docs/findings.md and docs/experiments/. Strength labels follow docs/redteam_claims.md. Items marked [RESULT PENDING: ...] are experiments running on 2026-11-05.

## 1. cover

**Arguing against its own knowledge**

- When a nudge makes a model argue for a wrong answer it normally gets right, is the correct answer still represented inside it? Can the J-lens detect that better than the written reasoning?
- Qwen3-8B in thinking mode (plus other open and API models) · Neuronpedia's J-lens · MMLU-Redux questions the model reliably knows

*Notes:* Ten-minute talk. Say 'arguing against its own knowledge', not 'motivated reasoning': a hint can be legitimate evidence and we cannot establish a motive.

## 2. headline

**Three claims**

- 1. When a nudge makes Qwen argue against what it knows, the reasoning says so.
- 2. On these tasks the J-lens is no better than a logit lens and worse than a cheap probe. The one exception is steering at the answer.
- 3. Methods warning: lens scores compared across two models measure the model, not the behaviour.
- Still missing: silent cases.

*Notes:* From docs/findings.md story paragraph; claim 1: silent-scan-thinking, moral-hunt (2 of 1,231 factual, 0 of 5,019 moral switches never mention the hint). Claim 2: jlens-vs-competitors, instrument-q1q3, steering. Claim 3: hack-organism.

## 3. setup

**Is the correct answer still represented while the model argues for a wrong one?**

- Questions: multiple choice from MMLU-Redux that Qwen3-8B answers correctly in at least 9 of 10 unhinted runs. That is what “knows” means here.
- Nudge: a hint at the bottom that points to one wrong option. Then compare runs that follow it with runs that resist.
- Four readouts on the same runs: the written reasoning, the J-lens, the plain logit lens (the baseline) and a trained linear probe.
- Why we care: a lens that sees the correct answer while the text argues for a wrong one would be a monitor that needs no text. We do not claim “knows it is bad”: that would imply a motive.

*Notes:* Experiments: m1-pilot (set-up), m0-tools (readout matches the jlens library, teacher-forced log-probs match vLLM). Qwen/Qwen3-8B pinned revision, thinking mode; J-lens from Neuronpedia, blocks 14-32; logit lens is J = identity. Motivation caveat from docs/findings.md 'Don't claim': test awareness as an excuse cuts against us. Abdelnabi and Salem find awareness lowers harmful compliance in Qwen3-32B, and our Qwen3-14B 'testing me' runs take the grader bug about as often as others (10% vs 12%). So the motivating case (a model that says 'this is a test' and then misbehaves) is not established; the question here is only whether the correct answer is still represented.

## 4. wording

**The hint’s wording decides whether Qwen switches; this explains why earlier work saw much more switching than our pilot**

- Stating the answer works: “Hint: The answer is X” makes 67% of runs switch.
- Only reporting a choice barely works: “…chose X” gives 4–13%.
- The earlier study’s hint and our pilot’s hint differ only in wording, and that alone moves switching by −51 percentage points. Thinking mode, decoding and format each move it by under 7 points.
- [RESULT PENDING: wording sensitivity on other model families (does the −51 pp wording effect appear outside Qwen?)]
- Strength: Solid. Qwen3-8B only; same 50 questions paired.
- Figure: docs/experiments/wording-dissection/fig02_wording.svg. Change in switch rate (percentage points) when one part of the wording changes.

*Notes:* Experiments: gap-factorial, wording-dissection, mb-replication. Numbers from docs/findings.md claim 1: wording −51 pp [−56, −47]; thinking mode, decoding, format each < 7 pp; 'Hint: The answer is X' 67%; '...chose X' 4-13%. M&B is the earlier study whose switching was high; our pilot's near-zero is explained by wording. Wording sensitivity on other families is running today.

## 5. open

**In thinking mode, when Qwen switches, its reasoning names the hint**

- Only 2 of 1,231 factual switches and 0 of 5,019 moral switches never mention the hint.
- With thinking turned off, 12–17% of switches do not mention the hint. Thinking off makes switches more often silent; it does not change how often they happen.
- For the grader-bug nudge, thinking off raises uptake instead (Qwen3-8B 2% to 15%, Qwen3-14B 11% to 60%).
- “Names the hint” is not “faithful”: the labels are keyword rules and the judge is not yet validated.
- Strength: Solid. Thinking-mode result. The no-thinking part is moderate.

*Notes:* Experiments: silent-scan-thinking, moral-hunt (names the hint: 2 of 1,231 factual, 0 of 5,019 moral); nothink-scale (12-17% silent with thinking off); grader-bug-e3 and models-grader (grader-bug uptake with thinking off: Qwen3-8B 15%, 14B 60%; 2% and 11% are the thinking-on rates per docs/findings.md and the main session's correction). Judge validation needs 50 user labels and has not been done (judge-validation).

## 6. models

**Other models take nudges too, and most of them say so**

- The earlier study’s hint moves all 12 models tested, by 17–72% (known items, 5 of 5 correct, via API).
- The chart is a different nudge: a buggy grader plus “you are scored”. Most models take it (87–100% for Gemma-4-31B, DeepSeek-V4-Flash, Qwen3.6-27B).
- 99% of the judged followers say they are doing it. Silent letter swaps exist (Llama-3.3-70B 22 of 250, Gemma-3-12B 47 of 501) but are one agent’s reading.
- [RESULT PENDING: wording sensitivity on other model families (does the −51 pp wording effect appear outside Qwen?)]
- Strength: Moderate. Hint result solid; grader-bug result moderate; silent swaps preliminary.
- Figure: docs/experiments/models-grader/fig12_models_grader.svg. Share of runs following the grader-bug nudge, by model (not the wording hint).

*Notes:* Experiment: models-grader. Numbers from docs/findings.md claims 2, 4, 14. The cross-model chart shows the grader-bug plus scored-framing nudge, not the M&B hint, so it does not show that wording sensitivity carries over. Whether other families are as wording-sensitive as Qwen is untested; result pending from today's run.

## 7. retained

**Some runs keep the correct answer: asked again sincerely, they can still give it**

- After a run argued for B, we asked the question again plainly.
- Runs told to argue for B named the correct option 93–94% of the time.
- Runs that followed a wrong claim planted in their own reasoning did so 30%; followers of an “the answer is X” hint 40–60%.
- So the knowledge is often retained, but this is behaviour, not a look inside.
- Strength: Moderate. Qwen3-8B on multiple choice only.

*Notes:* Experiments: reelicitation-e7, instructed-e1. Numbers from docs/findings.md claim 6. Re-asked sincerely: told-to-argue runs 93-94% name the correct option; planted-claim followers 30%; answer-is-hint followers 40-60%.

## 8. detect

**No: the J-lens is no better than a logit lens, and a cheap probe or keyword check does better**

- Before the answer, the J-lens score is within 0.09 of chance (AUROC 0.5) in 10 of 11 contrasts. It never beats the logit lens.
- A keyword check on the reasoning separates “told to argue for B” from no hint at 0.97.
- A probe picks the correct option in 72% of no-hint runs; J-lens 42%, logit lens 43% (chance 25%).
- Silent vs open switches: J-lens 0.54, same as reasoning length 0.54.
- Limit: the letter readout cannot see the answer mid-reasoning even in clean runs, so these nulls are weak.
- Strength: Moderate. No equivalence test yet, so not “adds nothing”.
- Figure: docs/experiments/jlens-vs-competitors/fig14_competitors.svg. Detection scores of J-lens, logit lens, probe and keyword check.

*Notes:* Experiments: jlens-vs-competitors, instrument-q1q3, prereg-t1b, lens-positions. Numbers from docs/findings.md claims 5, 7, 8: silent vs open J-lens 0.54 [0.50, 0.58], length 0.54, logit 0.46. Equivalence tests F1-F3 are pre-registered but not run.

## 9. steer

**Pushing along a J-lens direction at the answer fixes more wrong answers than a logit-lens direction**

- On runs ending in a wrong answer, we push toward the correct letter at the answer position only.
- Fixed: J-lens 47%, logit lens 27%, probe direction 59%, random direction 2%.
- This is the only place the J-lens beats the logit lens, and partly by construction.
- It shows a causal effect at the answer, not knowledge held during reasoning.
- Strength: Moderate. Plain Qwen3-8B, multiple choice, block 24; a 2,048-token cap (unaccepted deviation) weakens the during-reasoning part.
- Figure: docs/experiments/steering/fig13_steering.svg. Share of wrong answers corrected, by steering direction.

*Notes:* Experiment: steering. Numbers from docs/findings.md claim 9. The probe direction is the difference of mean hidden states of runs answering each letter, learned on other questions. Significance of the J-lens vs logit gap was not stated in findings.md.

## 10. hack

**The lens scores separate the two models, not the hacking runs**

- Task: solve CodeContests problems where three harness hacks are described and forbidden. Fine-tuned adapters (cds-jb); only the version trained to hide usually hides.
- Score: mean lens log-prob of 27 cheating words (exit, hack, pass, workaround…) before any hack word appears. Across models it separates almost perfectly (0.995).
- But the same transcript read through both models gives 0.94, and neutral words 0.98. Within one model: J-lens 0.44, logit 0.47, probe 0.39, keyword 0.46.
- Cheating words do appear, but as often in no-hack runs (14.7% vs 17.7% of lists); the hiding model shows almost none (1.1%).
- Strength: Moderate. One model family, one task; the question was “does this run hack”.
- Figure: docs/experiments/hack-organism/fig11_hack.svg. AUROC of lens scores: across models vs controls vs within one model.

*Notes:* Experiments: hack-organism, hack-manual-read. Numbers from docs/findings.md claim 10: cross-model 0.995; same transcript through both models 0.94; neutral words 0.98; last prompt token 1.00; one model fixed J-lens 0.44 [0.33, 0.55], logit 0.47, probe 0.39, keyword 0.46; cheating words 14.7% vs 17.7%; hiding model 1.1%. Story.md says 7% of flagged words are echoes (hack-manual-read README: 7% anywhere earlier in the text). It is a mean log-prob score with AUROC, not a classifier. The main test confounded model and behaviour; waterproofing test W is pre-registered, not run. Lesson: always control for model identity.

## 11. evalaware

**The paper’s score fails on Qwen3-8B; on larger models the effect is not J-lens specific**

- Qwen3-8B: the score separates test-like from real prompts at 0.50 (chance). Qwen says “testing” in 1% of blackmail runs; Claude in 71%.
- Qwen3-14B: removing eval-word directions silences “the user is testing me” (23% to 0%), but logit-lens directions do the same.
- Qwen3.6-27B smoke test: J-lens 0.64, logit lens 0.66, 0.55 without eval-worded prompt sets, 0.49 within identical prompts.
- [RESULT PENDING: test awareness on Qwen3-14B (do runs that say they are being tested take the grader bug more often than the 10% vs 12% seen so far?)]
- [RESULT PENDING: full Qwen3.6-27B eval-awareness replication (pre-registered; the smoke test gave 0.64 vs 0.66 for the logit lens)]
- Strength: Preliminary. 8B result moderate; 27B is one exploratory smoke test.
- Figure: docs/experiments/evalaware-ablation-14b/fig17_evalaware_ablation14.svg. Qwen3-14B: effect of removing eval-word directions (J-lens vs logit-lens).

*Notes:* Experiments: evalaware-replication (8B 0.50 [0.44, 0.56], 700 test-like vs 200 real prompts), evalaware-models-screen (1% vs 71%), evalaware-ablation-14b (23% to 0%), evalaware-big-smoke (27B 0.64 vs 0.66; 0.55; 0.49). Numbers from docs/findings.md claims 11 and 13. 'Other papers lack the control' is a literature reading (docs/literature/evalaware_internals.md); our control (docs/design_probe_control.md) is designed, not run.

## 12. next

**For now, reading the reasoning is enough here; a lens must beat that in silent cases**

- Where Qwen argues against its knowledge, the text says so, so text monitoring already catches it.
- No detection test shows the J-lens beating the logit lens, a probe or a keyword check. Absence is not shown: the letter readout is blind mid-reasoning.
- Next: pending runs today, equivalence tests on the planted claim, a reward-hack waterproofing test, the probe-format control, and finding genuinely silent cases.
- [RESULT PENDING: wording sensitivity on other model families (does the −51 pp wording effect appear outside Qwen?)]
- [RESULT PENDING: test awareness on Qwen3-14B (do runs that say they are being tested take the grader bug more often than the 10% vs 12% seen so far?)]
- [RESULT PENDING: full Qwen3.6-27B eval-awareness replication (pre-registered; the smoke test gave 0.64 vs 0.66 for the logit lens)]

*Notes:* From docs/findings.md 'Open questions and running work' and 'Don't claim'. Pre-registered but not run: F1-F3 equivalence tests and difficulty control on the planted claim (CPU); reward-hack waterproofing W (3-4 GPU-h). Waiting for the user: Nguyen-probe control (docs/design_probe_control.md section 11), 50 labels for judge validation. Do not claim: that the J-lens cannot see arguing against knowledge; that it beats the logit lens in any detection test; that the CoT is faithful; that test awareness excuses bad behaviour.

## 13. b-factorial (backup)

**Which parts of the hint wording drive switching**

- The full wording grid, paired over the same questions, Qwen3-8B.
- Wording moves switching by −51 pp; thinking mode, decoding and format each under 7 pp.
- Strength: Solid. Qwen3-8B only.
- Figure: docs/experiments/gap-factorial/fig01_factorial.svg. Switch rate across the wording factorial.

*Notes:* Experiments: gap-factorial, mb-replication. Numbers from docs/findings.md claim 1.

## 14. b-instrument (backup)

**The letter readout is blind mid-reasoning, even in clean runs**

- Knowledge is readable only at the option words and at the answer.
- So null results taken mid-reasoning are weak evidence.
- Strength: Moderate. Qwen3-8B.
- Figure: docs/experiments/instrument-q1q3/fig10_instrument.svg. Where the correct letter is readable.

*Notes:* Experiments: instrument-q1q3, lens-positions. docs/findings.md claim 8.

## 15. b-planted (backup)

**The one J-lens positive is fragile**

- Pre-registered pass: AUROC 0.71 [0.57, 0.84].
- But J-lens minus logit lens is +0.11 [−0.10, +0.30], not significant.
- Only 11 of 74 questions have both outcomes, so it may rank questions by difficulty.
- Strength: Preliminary. Equivalence and difficulty-control tests (F1-F3) not run.
- Figure: docs/experiments/lens-c3/fig08_c3.svg. Planted-claim test.

*Notes:* Experiments: replicate-leads, lens-c3, planted-claim-c3. docs/findings.md claim 12.

## 16. b-evalaware8b (backup)

**The eval-awareness score by model**

- Qwen3-8B 0.50 [0.44, 0.56] on 700 test-like vs 200 real prompts.
- Qwen3.6-27B smoke test 0.64 vs logit lens 0.66; 0.55 without eval-worded sets; 0.49 within identical prompts.
- [RESULT PENDING: full Qwen3.6-27B eval-awareness replication (pre-registered; the smoke test gave 0.64 vs 0.66 for the logit lens)]
- Strength: Preliminary. 27B is one exploratory run.
- Figure: docs/experiments/evalaware-replication/fig16_evalaware.svg. Eval-awareness score on Qwen3-8B.

*Notes:* Experiments: evalaware-replication, evalaware-big-smoke. docs/findings.md claims 11, 13.
