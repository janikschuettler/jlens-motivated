---
id: lens-main
title: 'Main lens readout: the answer position and the salience control'
strand: C
also: []
status: done
kind: exploratory
verdict: against
confidence: moderate
dates:
  start: 2026-11-03T00:52
  end: 2026-11-03T02:45
  approx: false
timing:
  wall: about 2 h
  gpu_hours: null
  api_usd: 0
  note: Readouts at 35–67 runs/min (thinking) and about 200/min (non-thinking); total GPU time not recorded. 14 GB for the thinking sets.
machine: arena9-mayor (RTX 4090)
n:
  items: up to 164 known (thinking), 149 (non-thinking)
  runs: 7,648
numbers:
- label: 'Told to argue for B: correct-letter margin (layers 30–32)'
  value: +1.4 to +1.9 nats
- label: Followed M&B's hint
  value: +1.6 to +2.0
- label: Followed a claimed answer key, repeat B when asked
  value: +2.5 to +2.9
- label: Honest errors (no hint)
  value: +0.7 to +1.7
- label: 'Salience control: the unchosen hint letter in runs that resisted'
  value: +0.5 to +2.1
- label: Probe at the answer position, runs that followed
  value: 'P(B) 0.97–1.0: it reads the final answer'
- label: 'Told-to-argue vs persuaded runs (AUROC): lens and probe / reasoning keywords / sincere follow-up'
  value: AUROC 0.35–0.57 / 0.47–0.56 / 0.97
- label: 'Lead: thinking off, silent vs open switches (AUROC)'
  value: J-lens 0.65 [0.56, 0.74], logit 0.48 [0.37, 0.58]
- label: 'Lead: claim followers who give the correct answer when asked vs those who repeat B (AUROC)'
  value: 0.78 [0.65, 0.92] (post-hoc layers, 10 vs 12 items)
numbers_note: 'Margins are in nats: how far the correct letter sits above the other unchosen wrong letters at the answer position (0 = no difference). A margin only counts as a sign of knowledge if it is clearly above the salience control. AUROC rows: 0.5 is chance, 1 is perfect.'
reports:
- report.md
figure: fig05_lens_margins.svg
data:
- data/lens_main.json
commits:
- 700e955
related:
- lens-positions
- prereg-t1
- prereg-t2
---

# Main lens readout: the answer position and the salience control

## Question

In runs that end on a wrong answer, is the correct option still represented, beyond what any prominent unchosen option keeps?

## Method

7,648 runs read out with the J-lens and logit lens at layers 14–32, on the saved token IDs.
Thinking-mode groups on known questions: told to argue for B [I1/I3]; followed M&B's hint; followed a claimed answer key, split by what they said when asked sincerely afterwards; honest errors on questions the model gets right only sometimes; and the matching resisted and no-hint runs. Thinking-off groups from the factorial.
Also a linear probe on the stored hidden states, with questions held out. Layers 30–32 were chosen after looking.

## Result

No. Every wrong-answer group keeps the correct letter 1–3 nats above the other wrong letters at the answer position. But the salience control keeps about as much (+0.5 to +2.1), so this is no sign of hidden knowledge.

## Interpretation

At the answer position, the correct letter keeps what any prominent unchosen option keeps. By itself that margin is not evidence that the model still holds the correct answer. The two small leads went on to pre-registered tests [T1, T2].

## Caveats

- The salience control is crude: the hint letter and the correct letter are prominent for different reasons.
- Groups have different prompts.
- Some groups have many runs per question.

## Files

- Report: [report.md](report.md)
- Figure: [fig05_lens_margins.svg](fig05_lens_margins.svg)
- Derived numbers: [data/lens_main.json](data/lens_main.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/lens/analyze_main.py`
- `scripts/lens/analyze_nothink.py`
