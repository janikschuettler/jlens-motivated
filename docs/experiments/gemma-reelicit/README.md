---
id: gemma-reelicit
title: 'Gemma-3-27B asked sincerely after a silent switch: does it name the right answer?'
strand: B
status: done
kind: preregistered
verdict: contradicts
confidence: moderate
dates:
  start: 2026-11-05T23:10
  end: 2026-11-05T23:40
  approx: true
timing:
  wall: about 30 min
  gpu_hours: 0
  api_usd: 0.04
  note: OpenRouter, DeepInfra, 959 one-token re-asks.
machine: arena9-cloud (OpenRouter API)
n:
  items: 40 known (5/5)
  runs: 959 re-asks (82 silent switches on 26 items)
numbers:
- label: Silent switches whose re-ask names the correct option
  value: 0.11 [0.04, 0.21]
- label: Open switches (reasoning names the cue), same measure
  value: 0.44 [0.32, 0.59]
- label: Resisted runs / no-cue runs, same measure
  value: 0.97 [0.95, 0.98] / 1.00
- label: Silent switches whose re-ask repeats the switched letter
  value: 0.89 [0.79, 0.96]
numbers_note: 'Brackets are 95% item-bootstrap CIs. Re-ask at temperature 0, one letter, whole original conversation resent as text.'
data:
- data/reask.json
related:
- gemma27-silent
- reelicitation-e7
- subtle-cues-cross-model
---

# Gemma-3-27B sincere re-ask after silent switches

## Question
Qwen3-8B runs told to argue for B name the correct option 93-94% when re-asked sincerely (reelicitation-e7). Does Gemma-3-27B, after a switch to a cue's wrong letter that never mentions the cue, name the correct option when re-asked?

## Method
Pre-registered ([preregistration.md](preregistration.md), committed before any re-ask). API runs of google/gemma-3-27b-it (DeepInfra) from subtle-cues-cross-model and wording-cross-model: 40 known items, 9 arms. To every parsed run we appended its own reply and a follow-up user turn (reelicitation-e7's wording plus "Setting aside any hints, notes or examples in the question above"), temperature 0, one letter. Groups: silent switch = answered the cue's wrong letter and the text matches none of the cue keywords (the loose keyword rule); open switch = same but it matches; resisted = answered correctly under a cue; no-cue = unhinted run. The conversation is re-sent as text (API).

## Result
Silent switches name the correct option only 11% [4, 21] of the time and repeat the switched letter 89% [79, 96] (82 runs, 26 items; mostly marked examples, 56, and "the answer is B", 17). Open switches: 44% [32, 59]. Resisted runs 97% [95, 98]; no-cue runs 100%. The pre-registered rule for "mostly still knows" fails (share well under 0.5). Silent vs open differ (CIs just barely do not overlap at the top of silent and bottom of open: 0.21 vs 0.32).

## Interpretation
Opposite of the Qwen result. In Gemma, a silent switch is not undone by a sincere question: it commits to the switched letter, so by behaviour there is no sign of hidden knowledge in these runs. Caveat that cuts either way: the re-ask sits after the model's own committed answer in context, and Gemma may simply stay consistent with it; Qwen's runs were instructed to argue, which is a different situation (told to defend B vs quietly absorbing a cue). Silent here is the loose keyword label (about 15% too high by eye). Not a test of internal representation: the lens result for Gemma (gemma27-silent) stands separately. Per-arm: marked examples 16% [7, 28]; "answer is B" 0 of 17 (9 items).

## Caveats
Single temperature-0 re-ask per run; re-ask with the reply kept in context only (no fresh-chat variant); keyword silence; item overlap across arms; DeepInfra only.

## Files
[data/reask.json](data/reask.json); script scripts/explore/gemma_reelicit.py; raw re-asks in data/subtle_cross_model/gemma-3-27b-it/reask.jsonl (gitignored).
