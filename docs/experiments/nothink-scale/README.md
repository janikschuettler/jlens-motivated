---
id: nothink-scale
title: Silent switches with thinking turned off
strand: B
also:
- C
status: done
kind: exploratory
verdict: supports
confidence: moderate
dates:
  start: 2026-11-03T02:36
  end: 2026-11-03T02:43
  approx: false
timing:
  wall: 7 min
  gpu_hours: 0.18
  api_usd: 0
  note: 3.6 GPU-min of generation plus about 7 min of readout (about 220 runs/min).
machine: arena9-octavia (L40S)
n:
  items: 149 known (M&B's setting, thinking off)
  runs: 1,490
numbers:
- label: Switch rate
  value: 710 / 1,192 hinted runs (60% [54, 65])
- label: Switches with no 'hint' or 'expert' anywhere
  value: 124 / 710 (17%)
- label: Switches with no broader reference to the hint either
  value: 101 / 710 (14%)
- label: 'Fresh questions (T1b): keyword-silent switches'
  value: 986 / 8,254 (12%)
- label: 'No hint: correct answers'
  value: 99.3% correct
numbers_note: null
reports:
- ../wording-dissection/report.md
- ../prereg-t1/report.md
figure: ../subtle-cues/fig04_silent_share.svg
data:
- ../wording-dissection/data/wave2_part2.json
commits:
- af18607
related:
- prereg-t1
- prereg-t1b
---

# Silent switches with thinking turned off

## Question

With thinking mode off, how many switches are silent?

## Method

The factorial's thinking-off condition with M&B's hint, format and decoding, on all 149 questions known in that setting × 8 samples, plus 2 unhinted samples. Keyword and broad scans; 10 silent switches read by eye. All runs were read out with the lenses for the pre-registered test T1.

## Result

Many more: 60% of runs switch, and 101-124 of the 710 switches (14-17%) never mention the hint. On 1,744 fresh questions (test T1b), 12% of switches are keyword-silent.

## Interpretation

Silent switches exist when the model doesn't reason in a separate thinking block. On reading, they argue for the hinted option on its merits; several call the correct option 'true' or 'almost correct' and then dismiss it. This is the main pool where a lens could beat the reasoning text.

## Caveats

- Keyword-silent means no hint word, not no sign of the hint's pull.
- Silent switching concentrates on a few questions (115 of 124 on 42 questions).

## Files

- Report: [../wording-dissection/report.md](../wording-dissection/report.md)
- Report: [../prereg-t1/report.md](../prereg-t1/report.md)
- Figure: [../subtle-cues/fig04_silent_share.svg](../subtle-cues/fig04_silent_share.svg)
- Derived numbers: [../wording-dissection/data/wave2_part2.json](../wording-dissection/data/wave2_part2.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `configs/explore/w2_nothink_scale.yaml`
