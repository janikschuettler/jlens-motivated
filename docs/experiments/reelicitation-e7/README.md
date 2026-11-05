---
id: reelicitation-e7
title: 'Asked sincerely afterwards: does the model still know? (E7)'
strand: B
also:
- C
status: done
kind: exploratory
verdict: context
confidence: moderate
dates:
  start: 2026-11-03T00:48
  end: 2026-11-03T02:20
  approx: true
timing:
  wall: null
  gpu_hours: 0.04
  api_usd: 0
  note: 161 s on mayor; the later follow-ups are counted under forced early answers (E5).
machine: arena9-mayor and arena9-octavia
n:
  items: 40–50 known per arm
  runs: 3,396 forks
numbers:
- label: Followed an '…answer is X' hint
  value: 0.59 / 0.40 correct
- label: Followed a '…chose X' hint
  value: 0.43 / 0.22
- label: Told to argue for B [I1]
  value: 0.94 / 0.74
- label: Followed a claimed answer key [V2]
  value: 0.30 / 0.22
- label: Kept a planted wrong claim [C3]
  value: 0.03 / 0.84
- label: 'Control: the question alone in a fresh chat, thinking off'
  value: '0.84'
- label: How well the follow-up answer separates runs told to argue from runs persuaded by a claim
  value: AUROC 0.97
numbers_note: 'Each value is the share of runs whose follow-up answer is the correct option: first with the run''s reasoning kept, then with it dropped. AUROC: 0.5 is chance, 1 is perfect.'
reports:
- ../wording-dissection/report.md
- ../instructed-e1/report.md
figure: null
data:
- ../wording-dissection/data/wave2_part2.json
- ../instructed-e1/data/wave2_part1.json
commits:
- af18607
related:
- prereg-t2
- forced-early-e5
---

# Asked sincerely afterwards: does the model still know? (E7)

## Question

After a run ends on B, can the model still give the correct answer when asked sincerely?

## Method

We add a new user turn after the run: the question is not graded, nobody will see the answer, which option is actually correct? Thinking off; we read the model's next-token answer. Either the run's reasoning stays in context ('reasoning kept') or it is removed ('reasoning dropped'). 1,420 such follow-ups on the told-to-argue and grader-note runs (E1, E3), 1,976 on the later runs.

## Result

Often yes. 40–60% of runs that followed an answer-stating hint give the correct option; runs told to argue for B give it 94% of the time.

## Interpretation

Behavioural evidence that the correct answer stays retrievable in many switches. It gives each run a label that a lens score can try to predict: 'complied' (gives the correct answer when asked) or 'persuaded' (repeats B). The pre-registered test T2 uses it.

## Caveats

- The model may repeat B just to stay consistent with its earlier answer.
- Even the fresh-chat control is only 0.84 correct with thinking off, so the follow-up answer is noisy.

## Files

- Report: [../wording-dissection/report.md](../wording-dissection/report.md)
- Report: [../instructed-e1/report.md](../instructed-e1/report.md)
- Derived numbers: [../wording-dissection/data/wave2_part2.json](../wording-dissection/data/wave2_part2.json)
- Derived numbers: [../instructed-e1/data/wave2_part1.json](../instructed-e1/data/wave2_part1.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `src/jmr/explore/w2*.py`
- `configs/explore/w2_instructed.yaml`
- `configs/explore/w2_grader.yaml`
