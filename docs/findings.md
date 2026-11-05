# Findings: current state

Last curated: 2026-11-05, 12:30 (claims sorted by how far they can be pushed in tomorrow's talk).

This is the project's working memory. It holds what we know **now**, not a history. A new chat starts from `CLAUDE.md` and this file. For details, grep `docs/log.jsonl` (chronology) and `docs/experiments/<id>/` (one folder per experiment: README with front matter and details, report, figure, derived numbers).

**How to keep it.** After every result, rewrite the affected claim in place: update its numbers and strength, merge it with related claims, and move it to "Ruled out" if it fails. Delete what is superseded; don't append dated notes. Keep the file under about 70 lines. Each claim names its supporting experiments by their folder id in `docs/experiments/` (after the →). The dashboard is rebuilt in batches.

Conventions: strengths follow `docs/redteam_claims.md` (solid, moderate, preliminary). "Known item" means at least 9 of 10 unhinted samples are correct (API models: 5 of 5). AUROC runs from 0.5 (chance) to 1 (perfect). Brackets are 95% item-bootstrap CIs.

## The story in one paragraph
A nudge makes Qwen3-8B argue against an answer it knows only when the hint *states* the answer, and in thinking mode the reasoning nearly always names the hint, so reading the chain of thought already catches it. The J-lens has not shown more than the logit lens, a trained probe or a keyword check on the text in any detection test; it beats the logit lens only for steering at the answer. Lens scores compared across models measure the model, not the behaviour. We still lack silent cases, the setting where a lens could beat the chain of thought.

## Solid
1. **The hint's wording drives switching** (Qwen3-8B). Changing only the wording (M&B's "Hint: An expert indicates the answer is X" vs ours "An expert who is right 90% of the time chose X") moves switching by −51 pp [−56, −47]; thinking mode, decoding and format each < 7 pp. Stating the answer works ("Hint: The answer is X" 67%), reporting a choice doesn't ("…chose X" 4–13%). → gap-factorial, wording-dissection, mb-replication
2. **The M&B hint moves every model tested** (17–72% on 12 models, 5-of-5 known items, API). Whether other families are as *wording*-sensitive as Qwen is untested (running 2026-11-05). → models-grader
3. **In thinking mode the reasoning names the hint when the model switches**: 2 of 1,231 factual and 0 of 5,019 moral switches never mention it. Say "names the hint", not "faithful": keyword labels, judge not validated. → silent-scan-thinking, moral-hunt

## Moderate (claim with the stated scope)
4. **Thinking-mode Qwen in chat follows statements of the answer, not information-free incentives**: grader-bug note 2% vs null 0%; claimed answer keys 19–24%. Not general: told they are scored, most other models take the bug (87–100% for Gemma-4-31B, DeepSeek-V4-Flash, Qwen3.6-27B) and 99% of judged followers say so. Thinking off raises uptake (Qwen3-8B 15%, 14B 60%). → grader-bug-e3, models-grader
5. **Thinking off makes switches silent more often** (12–17% of switches don't mention the hint), but the J-lens barely tells silent from open switches (0.54 [0.50, 0.58]), no better than reasoning length (0.54); logit lens 0.46. → nothink-scale, prereg-t1b
6. **Behaviour shows retained knowledge in some runs**: re-asked sincerely, runs told to argue for B name the correct option 93–94%; runs that followed a planted claim 30%; "answer is" hint followers 40–60%. → reelicitation-e7, instructed-e1
7. **The J-lens is not the best detector here.** Before the answer, the J-lens score is within 0.09 of chance in 10 of 11 contrasts and never beats the logit lens in paired tests. A keyword check on the reasoning separates "told to argue for B" from no-hint at 0.97; a probe at the option words picks the correct option in 72% of no-hint runs vs J-lens 42%, logit lens 43% (chance 25%). No equivalence test yet, so not "adds nothing". → jlens-vs-competitors, instrument-q1q3
8. **Instrument limit:** the letter readout can't see the correct answer mid-reasoning even in clean correct runs; knowledge is readable only at the option words and at the answer. Mid-reasoning nulls are weak evidence. → lens-positions, instrument-q1q3
9. **J-lens directions steer the answer, better than the logit lens** (plain Qwen3-8B, multiple choice, block 24): pushing toward the correct letter at the answer fixes 47% of wrong answers vs logit 27%, probe 59%, random 2%. Partly by construction; a 2,048-token cap (unaccepted deviation) weakens the during-reasoning part. Causal at the answer, not evidence of knowledge held during reasoning. → steering
10. **Reward-hack model (cds-jb adapters, coding task): lens scores detect the adapter, not the hack.** Cross-model main test 0.995, but the same transcript read by both models gives 0.94, neutral words 0.98, the last prompt token 1.00. With one model fixed, hidden-hack vs no-hack: J-lens 0.44 [0.33, 0.55], logit 0.47, probe 0.39, keyword 0.46. By eye, cheating words (cheat, 作弊, bypass) appear as often in no-hack runs (14.7% vs 17.7% of lists) and sit before "I'll …" announcements; the hiding model shows almost none (1.1%). The J-lens amplifies the model shift more than the logit lens (same text 0.94 vs 0.59). → hack-organism, hack-manual-read
11. **The J-lens paper's eval-awareness score fails on Qwen3-8B** (700 test-like vs 200 real prompts: 0.50 [0.44, 0.56]); Qwen rarely says it's being tested on their blackmail prompt (1% vs Claude 71%). On Qwen3-14B the paper's causal effect reproduces (removing eval-word directions silences "the user is testing me", 23% → 0%), but logit-lens directions do the same: not J-lens specific. → evalaware-replication, evalaware-models-screen, evalaware-ablation-14b

## Preliminary (present as leads)
12. **Planted claim, the one J-lens positive:** pre-registered pass 0.71 [0.57, 0.84], but J − logit +0.11 [−0.10, +0.30] (n.s.) and only 11 of 74 questions have both outcomes, so it may rank questions by difficulty. → replicate-leads, lens-c3
13. **Qwen3.6-27B eval awareness** (one exploratory smoke test): 0.64, logit 0.66; 0.55 without the eval-worded prompt sets; within identical prompts before "testing me" 0.49. → evalaware-big-smoke
14. **Silent letter swaps** in Llama-3.3-70B (22 of 250 followers) and Gemma-3-12B (47 of 501): one agent's reading, few questions, API. The only silent behaviour found. → models-grader
15. **Published eval-awareness probes may read prompt format** (literature only; our control is designed, not run). → docs/literature/evalaware_internals.md, docs/design_probe_control.md

## Don't claim
- That the J-lens can't see arguing against knowledge (absence untested; letter readout limit, claim 8).
- That the J-lens beats the logit lens in any detection test.
- That the chain of thought is faithful (only that it names the hint).
- That test awareness excuses bad behaviour: Abdelnabi & Salem find it lowers harmful compliance; our 14B "testing me" runs take the bug about as often (10% vs 12%).
- Anything beyond Qwen and MMLU-Redux for the lens results.

## Open questions and running work (2026-11-05)
- Running: wording sensitivity on other families (API); test awareness on Qwen3-14B (generation done on derpy, 4,784 runs; gate next); Qwen3.6-27B eval-awareness replication (pre-registered EA27); slides; red team of tomorrow's story.
- Waiting for the user: the Nguyen-probe control (design ready, 9 decisions in docs/design_probe_control.md §11).
- Not run: equivalence tests and the difficulty control on the planted claim (pre-registered F1–F3, CPU); reward-hack waterproofing (pre-registered W, 3–4 GPU-h); judge validation (needs 50 user labels); Gemma-3-12B silent swaps under a lens (needs an HF token).
