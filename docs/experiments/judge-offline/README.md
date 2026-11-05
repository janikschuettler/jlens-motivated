---
id: judge-offline
title: LLM judge for the reasoning, built and checked offline
strand: T
also:
- B
status: done
kind: tooling
verdict: inconclusive
confidence: low
dates:
  start: 2026-11-02T21:35
  end: 2026-11-02T21:51
  approx: true
timing:
  wall: null
  gpu_hours: 0
  api_usd: 0
  note: CPU only.
machine: arena9-cloud (no GPU)
n:
  items: null
  runs: 50 gold-labelled runs
numbers:
- label: Keyword baseline vs reference labels
  value: 46/50, κ 0.89
- label: API calls made
  value: 0 (the OpenRouter key failed overnight; $0 spent)
numbers_note: null
reports:
- ../../log.jsonl
figure: null
data: []
commits:
- 25479d1
related:
- judge-validation
---

# LLM judge for the reasoning, built and checked offline

## Question

Can an LLM judge label whether the reasoning mentions the hint, and whether it leans on it, better than our keyword lists?

## Method

Code in envs/judge, src/jmr/judge.py and scripts/judge.py. An LLM judge (via OpenRouter) answers four questions about each run: does the reasoning mention the nudge, what is its stance on it, does it argue for B, and what is the final answer. It quotes its evidence. Code then sorts the run into one of four cases: no mention; mentions and dismisses it; mentions, relies on it and argues for B; no mention but argues for B. It has a cache and a dry-run mode. Checked without the API: our keyword baseline against 50 reference labels written by a Claude agent.

## Result

Built, not yet run. The keyword baseline agrees with the reference labels on 46 of 50 runs (Cohen's κ 0.89, where 1 is perfect agreement and 0 is chance).

## Interpretation

Every 'mentions the hint' and 'silent' label so far comes from keyword rules. On a small set they agree well with reference labels. The judge that would check them at scale has not run, so treat those labels as provisional.

## Caveats

- Four code fixes were pending before first use.
- The reference labels come from one Claude agent, not from humans.

## Files

- Report: [../../log.jsonl](../../log.jsonl)

Code and configs named in the text above:

- `scripts/judge.py`
- `src/jmr/judge.py`
