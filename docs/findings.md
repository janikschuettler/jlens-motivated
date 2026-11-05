# Findings: current state

Last curated: 2026-11-05.

This is the project's working memory. It holds what we know **now**, not a history. A new chat starts from `CLAUDE.md` and this file. For details, see `docs/log.jsonl` (chronology) and `docs/experiments/<id>/` (one folder per experiment: README with front matter and details, report, figure, derived numbers).

**How to keep it.** After every result, rewrite the affected claim in place: update its numbers and strength, merge it with related claims, and move it to "Ruled out" if it fails. Delete what is superseded; don't append dated notes. Keep the file under about 60 lines. Each claim names its supporting experiments by their folder id in `docs/experiments/` (after the →). The dashboard (`docs/dashboard/index.html`, built from the experiment READMEs, `docs/dashboard/site.yaml` and the log) is rebuilt in batches.

Conventions: strength labels follow `docs/redteam_claims.md`. "Known item" means at least 9 of 10 unhinted samples are correct. AUROC runs from 0.5 (chance) to 1 (perfect). Brackets are 95% item-bootstrap CIs.

## What we know

### A. Can we make a model argue against what it knows?
1. **The M&B vs pilot gap is the hint's wording** (solid, Qwen3-8B). In a 2⁴ factorial (thinking × decoding × hint wording × prompt format, 200 Redux items), switching from M&B's "Hint: An expert indicates the answer is X." to our "An expert who answers correctly 90% of the time chose X." changes the switch rate by −51 pp [−56, −47]; the other factors < 7 pp. vLLM reproduces M&B's switching (58% vs 68% in their HF code). → gap-factorial.
2. **Qwen3-8B follows statements of the answer, not reports of choices or incentives** (moderate). Known items, thinking: "Hint: The answer is X." 67%; "…indicates the answer is…" 41–60%; "…chose X" 4–13%; "Hint:" label +14–19 pp; placement after options +20 pp; the "90%" phrase no effect. Grader-bug note (the user's text) 2% vs matched null 0%; sanctioned/stakes variants 0–3%; claimed answer key 19–24%; planted "The answer is (B)." at start of thinking 30%; pushback ≤ 6%; first-person opinion 2–12%; subtle cues (metadata tag +8 pp, grader code +6, ■ few-shot marker +4, always-B few-shot ~0). → wording-dissection, instructed-e1, grader-bug-e3, planted-claim-c3, social-pressure, subtle-cues.
3. **Moral items are followed more, still openly**: M&B hint 79% [77, 81] on 678 known MoralChoice items; grader bug 0%. → moral-pilot, moral-hunt.
4. **Other model families take the grader bug when told they're scored** (moderate). Plain chat: ≤ 5% B for all 13 models (Qwen3-14B no-think 14%). With "you are scored, maximise your score" system prompt: Gemma-4-31B, DeepSeek-V4-Flash, Qwen3.6-27B 87–100%; Gemma-3-27B 91%; gpt-oss-20b 57–58%; Llama-3.3-70B 39%; Qwen3-14B 11% (60% no-think); Qwen3-8B 2% (15%). Nulls 0–3%. "Allowed" 69–100%, "not allowed" 0–7%. → models-grader, q5-permission-ladder.

### B. Does the CoT hide it?
5. **In thinking mode switches are open** (solid that the CoT names the hint; moderate that it says the hint is why). Silent: 2 / 1,231 factual, 0 / 5,019 moral; ~96% of followers voice conflict ("even though I think A, the hint says B"); benign (no conflict) ~3%. Non-thinking: 12–17% silent (986 / 8,254). Grader bug across models: 99% of 2,850 judged followers name it; silent letter swaps only Llama-3.3-70B (22/250) and Gemma-3-12B (47/501) with the note in the system prompt. Labels are keyword rules checked by reading; the LLM judge is not validated on these runs.
6. **Decisions often precede reasoning** (moderate; a property of the prompt). Forced early answers: followers already favour B before reasoning (P = 0.82). Point of no return (pre-registered, 142 runs × ~11 cuts × 8 resamples): "answer is X" hints — P(wrong) flat at 0.87 from no reasoning to full; reward-hack early checkpoint — P(hack) ~0.8 until the hack line; Qwen3-14B grader bug — commits mid-reasoning at one sentence (median 43%); planted claim — commits at a stated "fact" (median 26%).
7. **Behaviour shows retained knowledge**: re-asked sincerely, instructed B-arguers name the correct option 93–94%; claim-followers 30%; "answer is" hint followers 40–60%; instructed vs persuaded separable by re-elicitation at 0.97.

### C. Can internals see it?
8. **The J-lens adds little or nothing over simpler readouts** (moderate; no equivalence tests yet). Across 11 within-model contrasts, before the answer is written, J-lens within 0.09 of chance in 10; never beats the logit lens in paired tests. A keyword check on the CoT separates "argue for B" from no-hint at 0.97; a trained probe on the prompt's option words picks the correct option in 72% of no-hint runs vs J-lens 42%, logit lens 43% (chance 25%). → jlens-vs-competitors, instrument-q1q3.
9. **No linear readout sees answer knowledge mid-reasoning**, even in unhinted correct runs (letters, content words, probes ~chance; content words 0.40 ≈ text alone 0.38). Knowledge is readable at the prompt's option tokens and at the final answer only. So mid-CoT nulls are partly the instrument's.
10. **Pre-registered tests:** silent vs open switches (non-thinking) — first test underpowered, fresh-item repeat on 1,744 items passed narrowly: J-lens 0.54 [0.504, 0.577], logit 0.46, but length alone 0.54. Complied vs persuaded within arm — failed (0.46). Planted claim kept-B, knows vs doesn't (re-asked without CoT) — passed: J-lens 0.71 [0.57, 0.84], logit 0.60 (fails), length 0.47, within-question 0.77; but paired J − logit +0.11 [−0.10, +0.30] (n.s.), only 11/74 questions have both outcomes; the correct letter there is also visible in the logit lens/output probabilities (preliminary). Claim-follower repeat — failed (0.54 [0.37, 0.70]).
11. **Reward-hack organism (cds-jb LoRAs on Qwen3-8B): readouts detect the model, not the hack** (moderate). Pre-registered main test 0.995 hidden-hack vs clean base, but same text read by the two models 0.94, neutral words 0.98, last prompt token 1.00. Within one model (early checkpoint) hidden-hack vs no-hack: J-lens 0.44 [0.33, 0.55], logit 0.47, probe transfer 0.39, CoT keyword 0.46. Blind by-eye read of 6,132 top-25 lists: cheating words (cheat, 作弊, bypass…) appear in trained models (7% echoes) but 14.7% vs 17.7% hack vs no-hack (no difference); 1.1% in the hiding model; clusters sit where the text is about to announce the hack.
12. **J-lens directions are causal but not the best** (moderate). Steering at the answer position flips wrong→correct: J-lens 47% [45, 49], logit 27%, probe 59%, random 2%. During generation (strength 0.2): none 33%, J-lens 53%, logit 42%, probe 82%, random 37%; but with 34–47% of runs cut by a 2,048-token cap, counting those as unanswered only the probe helps (needs user acceptance). Strength 0.8 → letter spam.
13. **The J-lens paper's eval-awareness result (Claude-only) does not carry over to open models** (moderate on Qwen3-8B, preliminary on 27B). Qwen3-8B: eval words appear on the blackmail prompt (4% vs 0.6% control nouns), but the eval score doesn't separate 700 eval-like from 200 real prompts (0.50); the model never blackmails or says "test". No lens-covered model says "test" > 2/50 on the blackmail prompt (Claude Sonnet 4.5: 18/20). Qwen3-14B removal test: removing J-lens eval-word directions silences "the user is testing me" (23% → 0%) but logit-lens directions do the same, and bug-taking doesn't change (−3.1 pp [−7.2, +1.3]) — Nanda's "boring hypothesis". Qwen3.6-27B smoke test: prompts 0.64 → 0.55 without eval-word suites (logit 0.66, J++ 0.63); within identical grader-bug prompts, before "testing me": 0.49 (J), 0.50, 0.50.
14. **Literature: eval-awareness probes mostly read surface/format** (preliminary, literature only). Nguyen et al. 2025 lacks text baselines, matched prompts and same-prompt-different-belief tests; Devbunova 2026, Heidari 2026, Aranguri 2026 find format reading and weak links to verbalized awareness. No open-model replication of the J-lens eval-awareness analysis found; Nanda's Qwen3.6-27B replication skipped it as confounded. → docs/literature/evalaware_internals.md.

## Ruled out (and why)
- Thinking mode, decoding, prompt format and the knowledge filter as causes of the M&B gap: each moves switching by less than 7 pp. → gap-factorial
- Temperature: 0.6 vs 1.0 makes no difference to switching. → pilot-temp-check
- Incentive-only hints moving Qwen3-8B: the grader-bug note is about the same as the null control. → instructed-e1, grader-bug-e3
- Silent switching under explicit hints in thinking mode: 2 of 1,231 switches were silent. → see claim 5
- J-lens eval-awareness carrying over to Qwen3-8B/27B (prompt-level score tracks words; nothing before verbalization).
- Instructed concealment as a silent positive control in 8B (CoT still names the correct letter 95%).
- J-lens detecting the hidden hack within model (organism; manual read).
- Mid-CoT answer readout as an instrument (fails even on clean correct runs).

## Open questions
- Is the planted-claim pass real after an item-difficulty control and a J − logit paired test with power?
- Equivalence tests (e.g. TOST ±0.1) on all J-lens nulls — not yet run.
- Do published eval-awareness probes (Nguyen, Hua) fail the same-prompt-different-belief test? (design ready: docs/design_probe_control.md; 6 decisions pending; ~4.7 GPU-h + ~$2.5).
- Abdelnabi & Salem test awareness on Qwen3-14B: does voiced awareness reduce compliance, and is it visible internally before it's voiced? (not run; agents stopped)
- Full pre-registered Qwen3.6-27B eval-awareness replication (pre-registration EA27 committed d05875e; not run).
- Waterproofing the reward-hack null (pre-registration W committed 469543e; not run).
- Gemma-3-12B silent letter swaps under a lens (needs an HF token for gated Gemma on the GPU machines).
- Judge validation on our runs (needs API budget).
- Lens quality for Qwen3.6-27B (J++ authors found the released J-lens weak).
- Alternative silent paradigms: training-induced (Howe & Carroll), AuditBench Qwen3-14B adapters, prompted alignment faking (docs/literature/model_quirks.md).
