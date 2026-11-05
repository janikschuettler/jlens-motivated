---
id: wording-dissection
title: Which part of the hint makes the model switch
strand: A
also:
- B
status: done
kind: exploratory
verdict: supports
confidence: moderate
dates:
  start: 2026-11-03T00:55
  end: 2026-11-03T01:42
  approx: false
timing:
  wall: 46 min
  gpu_hours: 0.77
  api_usd: 0
  note: 1,110 tokens/s, mean 2,560 tokens.
machine: arena9-octavia (L40S)
n:
  items: 50 known
  runs: 1,200
numbers:
- label: '"…indicates the answer is X" instead of "…chose X", both with "Hint:"'
  value: +47 pp [+36, +57]
- label: Same, without the "Hint:" label
  value: +37 pp [+26, +49]
- label: Adding the "Hint:" label to "An expert indicates the answer is X."
  value: +19 pp [+10, +28]
- label: 'Adding "who answers correctly 90% of the time" to "Hint: An expert chose X."'
  value: −7 pp [−12, −2]
- label: 'Naming an expert as source vs no source (M&B''s hint vs "Hint: The answer is X.")'
  value: −7 pp [−15, +0]
- label: M&B's hint after the options instead of before the question
  value: +20 pp [+10, +30]
- label: Switches whose reasoning names the hint
  value: all but 3
numbers_note: Each row is the change in switch rate, in percentage points (pp), when one part of the wording changes, paired over the same 50 questions.
reports:
- report.md
figure: fig02_wording.svg
data:
- data/wave2_part2.json
commits:
- af18607
related:
- gap-factorial
- prereg-t2
- silent-scan-thinking
---

# Which part of the hint makes the model switch

## Question

Which part of M&B's hint makes the model give up an answer it knows: the "Hint:" label, the claim "the answer is X", the expert as source, the stated 90% reliability, or where the hint sits?

## Method

50 known questions × 3 samples × 8 new hint wordings, thinking mode, our setting. Plus the factorial's two wordings (M&B's and ours) on the same questions, 2 samples each. The hint goes between the options and the answer instruction, unless marked 'before the question'. Each comparison is paired over the same questions.

## Result

What matters is whether the hint states the answer. "Hint: The answer is X." makes the model switch in 67% of runs; "An expert chose X." gets 4–13%. The "Hint:" label and placing the hint last add to it; naming an expert or its reliability doesn't help.

## Interpretation

Our pilot hint failed because it reports one person's choice, which the model weighs as an opinion against its own. The switches are open deference to what looks like the answer key. These runs also supply the pre-registered complied-vs-persuaded test [T2].

## Caveats

- 3 samples per wording.
- Hint mentions are a keyword check.
- One set of 50 questions.

## Files

- Report: [report.md](report.md)
- Figure: [fig02_wording.svg](fig02_wording.svg)
- Derived numbers: [data/wave2_part2.json](data/wave2_part2.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `configs/explore/w2_wording.yaml`
