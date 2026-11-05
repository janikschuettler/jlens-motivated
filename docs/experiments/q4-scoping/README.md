---
id: q4-scoping
title: Do open models cheat on their own and argue it away? (literature, Q4)
strand: N
also: []
status: done
kind: exploratory
verdict: against
confidence: moderate
dates:
  start: '2026-11-03'
  end: 2026-11-03T16:01
  approx: false
timing:
  wall: null
  gpu_hours: 0
  api_usd: 0
  note: Reading only.
machine: arena9-cloud (no GPU)
n:
  items: null
  runs: 72 public transcripts coded
numbers:
- label: 'Stock Qwen3-8B: AISI task / HackTrace / djinn'
  value: 0 of 192 / 0.9% / exploit rate 0.009
- label: 'Qwen3.8-27B cheating runs: reasoning names it / argues it away / unclear'
  value: 25 / 6 / 1 of 32
- label: Final message discloses the workaround
  value: 10 of 32
- label: 'OLMo-3.1-32B-SFT: its brief reasoning never mentions the hack'
  value: 28 of 34 (hack open in the code)
numbers_note: null
reports:
- report.md
figure: null
data: []
commits:
- 617c73e
related:
- hack-organism
- impossiblebench-pilot
- q5-permission-ladder
---

# Do open models cheat on their own and argue it away? (literature, Q4)

## Question

Do open reasoning models that have a Neuronpedia J-lens cheat on their own and justify it, instead of naming it?

## Method

Reading only: ImpossibleBench, Model Forensics, the AISI CodeContests task, djinn, HackTrace, School of Reward Hacks and others, checked against the list of lens files. One reader coded two public transcript sets: all 32 cheating runs of Qwen3.8-27B on ImpossibleBench-LiveCodeBench, and the 40 earliest hacks of OLMo-3.1-32B-SFT in AISI's task.

## Result

Not in any lens-covered model that fits our GPU. Stock Qwen3-8B barely cheats (0–1%). Where open models do cheat, their reasoning mostly says so.

## Interpretation

Moving the core question to rule-breaking needs a pilot first, most likely multi-turn ImpossibleBench with Qwen3-14B and Qwen3-8B. Meanwhile a Qwen3-8B trained to hack (setup 1) can calibrate the lens.

## Caveats

- The counts come from models one size or generation away from the lens-covered ones.
- One coder, keyword-selected excerpts.
- Some sources read only at abstract or model-card level.

## Files

- Report: [report.md](report.md)
