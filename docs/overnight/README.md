# Overnight exploration, 2026-11-02 → 03

Exploratory work while the user is away. Report page due 08:00. None of this changes `docs/plan.md` or the roadmap; those stay the user's decision. Exploration runs may go beyond MMLU-Redux where noted (the user asked about normative questions); each report says which data it used.

## Questions

1. What is a good experimental design to pin down "arguing against its own knowledge" (motivated reasoning, colloquially)? Factual vs normative/behavioural questions; hints that carry no information but give an incentive (grader-bug hint); alternatives to multiple-choice + hint.
2. What explains the gap between M&B (≈65% switches) and our pilot (≈0%)? Which factors are necessary for the model to follow a misleading hint?
3. Can the J-lens (or a probe) detect that the correct answer is still represented when the chain of thought doesn't show it?

## Machines tonight

- **arena9-octavia** (L40S): generation experiments.
- **arena9-mayor** (4090): allowed tonight only (not a standing resource; don't add it to CLAUDE.md). Lens readouts and overflow generation.
- One agent owns one GPU at a time. The orchestrator assigns them.

## Layout

- Code: `src/jmr/explore/` (gen env), `scripts/explore/` (thin CLIs), lens-side additions in `src/jmr/lens/` and `scripts/lens/`.
- Raw data on the GPU machine: `data/explore/<experiment>/` (gitignored).
- Derived reports: `docs/overnight/reports/<experiment>.md`; numbers for the web page: `docs/overnight/data/<experiment>.json` (small, derived only; no CoTs, no token IDs).
- Log: `docs/overnight/log.md`, one short entry per finished step (time, what, result, where).

## Rules for tonight

CLAUDE.md applies. In particular rule 1: on a failure, stop that experiment, write the command, error and options in the log, and hand back; don't use the listed workarounds. Rule 3 for any readout. No commits (the user didn't ask for them tonight).
