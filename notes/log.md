# Session log (2026-11-02 → 2026-11-05), handoff

Detailed per-night logs: `docs/overnight/log.md` (all nights). This is the decision-level account.

## Day 1 (2026-11-02, before the overnight runs)
- Wrote CLAUDE.md/roadmap from docs/plan.md; reframed to v1 ("arguing against its own knowledge"). M0 tools built and passed (lens equals `lens.apply` exactly; vLLM vs HF log-probs median 0.0006 nats; bf16 top-k ties forced a tie-aware check). Switched data to MMLU-Redux 2.0 only.
- M1 smoke tests: weak nudges gave ~0 switches. Temperature check (0.6 vs 1.0): no effect. Decided to keep T 0.6.
- M&B mini-replication with their own code: ~65% switching, mostly open deference. So the collapse came from our generation setup or wording.
- Moved GPU work from mayor (4090) to octavia (L40S). User: remove mayor from docs; mayor only when allowed per night.
- Committed (M0, M1, temperature check, literature/replication).

## Night 1 (2026-11-02 21:30 → 11-03 ~06:00)
- Session restarted mid-launch (moved to tmux); first wave of agents lost; relaunched. Old VS Code session was still alive and had to be told to stand down (duplicate agents briefly).
- OpenRouter key looked broken (401); user said pause it. Later found the agent process held a stale copy; a fresh shell works.
- Gap factorial (octavia): hint wording is the factor (−51 pp). Wording dissection: statements of the answer move the model; "chose X" doesn't.
- Bug: `known_items.jsonl` overwrote the `correct` letter with counts → 820 runs invalidated and rerun; loader now refuses non-letter `correct`.
- Wave 2 (mayor tonight, octavia): instructed groups, grader-bug matrix (incentive ≈ no effect), planted claims, pushback, opinion, moral pilot, forks.
- Lens pipeline built and validated; lens_main mostly null; non-thinking silent-vs-open lead 0.65 → pre-registered T1 (underpowered), T2 (failed), T1b on 1,744 fresh items (narrow pass 0.54, length matches it).
- Moral hunt: 79% follow, 0 silent. Report page published (artifact 7jWuFKZjenAmqqDMTokVEX).
- Day after: user asked for plainer writing, commit autonomy, a running findings record (markdown → then a dashboard). Ran concept scan (nothing), C3 lens analysis (only answer-position signal, also in logit lens), subtle cues (barely move), Q1/Q3 instrument (no mid-CoT readout works; option-token probe does), Q4 scoping (no published spontaneous silent cheating in runnable lens-covered models), setup 1 reward-hack organism (readouts detect the adapter, not the hack), models × grader bug (other families cheat openly when scored; $4.91 API).
- Dashboard built (artifact BZHUWiBPLb7zN7mzXXbvzy), rewritten in plain English after the user couldn't follow labels like H1/β/warm-start; CLAUDE.md writing rules added.

## Night 2 (2026-11-04 00:00 → 09:00)
- Goal: evidence on whether the J-lens is a good/bad readout; replicate J-lens eval awareness and find where it breaks.
- Results: J-lens vs competitors (at chance; never beats logit lens); eval-awareness replication on Qwen3-8B breaks at the first quantitative step (paper is Claude-only); steering (J-lens causal, probe better; token-cap deviation); models screen (no open model says "test" on the blackmail prompt); point of no return; Qwen3-14B removal test (silences talk, logit lens too, behaviour unchanged); planted-claim repeat passed (0.71); claim-follower repeat failed.
- GPU scheduling across agents on one L40S was the bottleneck; steering waited ~2 h. Summary page published (JjncJAxwMdWecdnGmuwpSm).

## Day 2 (2026-11-04)
- User allowed re-tokenizing API text (with fidelity check) → CLAUDE.md exception. 27B smoke test via API + CPU offload: no J-lens eval awareness before verbalization.
- Research: Nguyen et al. probe lacks key controls; no open-model J-lens eval-awareness replication; Nanda skipped it as confounded. Quirks catalogue (Abdelnabi & Salem on Qwen3 the top candidate).
- Blind manual read of reward-hack lens lists: cheating words present, not run-specific.
- derpy added as second GPU (rest of project); envs built; downloads started. User can request bigger GPUs.

## Night 3 (2026-11-04 18:00 → aborted)
- Launched six agents (presentation, red team, probe-control design, Hawthorne 14B on derpy, hack waterproofing on octavia, 27B replication). All eight were stopped at 18:09 by a user-side action (likely an interrupt when typing `source …` into Claude Code); relaunched ~20:57 (much later than I assumed — I misjudged the clock).
- Completed: red team (97cf199), presentation deck (artifact DmcYpY45W8qex2id6yqNrA, slides.md c51c7e9/ca6428f), probe-control design (eebc782), 27B pre-registration EA27 (d05875e), hack waterproofing pre-registration W (469543e).
- At ~22:10 the account hit its weekly usage limit (resets 2026-11-08 19:00 Europe/London); the GPU agents (Hawthorne, waterproofing, 27B, claims-firming) died before running GPU work. 79% of usage was general-purpose subagents (user's /usage note). No GPU experiment from night 3 produced results.

## Surprises / dead ends
- vLLM lacks `no_repeat_ngram_size`; M&B switching reproduced anyway.
- Agents' clocks vs mine: log timestamps sometimes inconsistent.
- `sync_to_gpu.sh` `data/` exclude also skipped `docs/overnight/data/` (fixed, anchored).
- Checksum compare initially failed due to locale sort order.
- Recurrent "waiting on background work" notifications from finished agents; had to TaskStop them.
