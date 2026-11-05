---
id: jlens-vs-competitors
title: The J-lens against cheaper readouts, on every case we have read out
strand: C
also:
- T
status: done
kind: exploratory
verdict: against
confidence: moderate
dates:
  start: 2026-11-04T00:05
  end: 2026-11-04T01:35
  approx: true
timing:
  wall: about 1 h 30 min
  gpu_hours: 0
  api_usd: 0
  note: 'CPU only: feature extraction minutes per set; AUROCs and probes about 29 min on octavia''s CPU (8 processes); option words 283 s.'
machine: arena9-octavia (CPU) and arena9-cloud
n:
  items: 21–994 questions per group; 59 hack problems
  runs: about 11,600 runs across 11 contrasts; 1,639 runs at the option words
numbers:
- label: 'Option words, share of no-hint runs with the correct option on top: probe / J-lens / logit lens'
  value: 72% [64, 78] / 42% [35, 49] / 43% [35, 51]
- label: 'Option words, known questions vs honest errors (AUROC): probe / J-lens / logit lens'
  value: 0.80 [0.70, 0.89] / 0.61 [0.48, 0.75] / 0.53 [0.36, 0.71]
- label: 'Told to argue for B vs no hint: CoT keyword monitor / J-lens mid reasoning'
  value: 0.97 [0.94, 0.99] / 0.42 [0.34, 0.49]
- label: 'Hint followers vs resisters, mid reasoning: probe / logit lens / J-lens'
  value: 0.68 [0.59, 0.76] / 0.61 / 0.50
- label: 'Hidden hack vs no hack, end of reasoning: probe / J-lens / logit lens'
  value: 0.69 [0.60, 0.79] / 0.57 [0.46, 0.68] / 0.54
- label: 'Silent vs open switches, answer position: J-lens / logit lens / model''s own answer'
  value: 0.54 [0.50, 0.59] / 0.43 / 0.37 (0.63 the other way)
- label: Largest J-lens effect the intervals allow before the answer, per case
  value: 0.58–0.75
- label: 'Only J-lens lead: kept-claim runs that still know vs not (post hoc, 10 negative runs)'
  value: 0.86 [0.77, 0.96] vs logit 0.77
numbers_note: 'AUROC: the chance that a run of the arguing-against-knowledge or cheating group scores higher than a comparison run; 0.5 is chance, 1 is perfect, values below 0.5 separate in the other direction. Option-word rows are shares of runs (chance 25%). 95% intervals resample questions.'
reports:
- report.md
figure: fig14_competitors.svg
data:
- data/jlens_vs_competitors.json
commits:
- '3878973'
related:
- jlens-option-tokens
- instrument-q1q3
- lens-main
- prereg-t1b
- prereg-t2
- lens-c3
- hack-organism
- models-grader
---

# The J-lens against cheaper readouts, on every case we have read out

## Question

On the runs we already have, does the J-lens tell runs that argue against the model's knowledge (or cheat) from their comparison runs better than cheaper readouts on the same runs? And does it see, at the prompt's option words, the answer knowledge a probe finds there?

## Method

No new generation; existing readouts, CPU only. Eleven within-model contrasts: told to argue for B vs no hint; hint followers vs resisters; followers who still know vs persuaded (pre-registered T2); silent vs open switches (thinking off); kept a planted wrong claim vs recovered, and kept-claim runs that still know vs not; hidden reward hack vs no hack in one model (warm-start adapter); grader-bug followers vs resisters in Qwen3-14B, gpt-oss-20b, and Qwen3-8B/14B with thinking off.
Positions: prompt option words, last prompt token, mid reasoning, end of reasoning, answer position.
Readouts: J-lens and logit lens (log p(correct letter) minus log p(pushed letter), workspace-layer mean; hack words for the hack); a linear probe trained on the two groups with questions held out; the model's own answer probabilities, forced early answers and re-asking where they exist; the CoT keyword monitor; response length.
Metric: AUROC with 2,000 question-bootstrap resamples; paired bootstrap for J-lens minus logit lens. Earlier reports' numbers reused where they exist, the rest computed.
Option words: the J-lens and logit lens applied on the CPU to the same stored option-text hidden states the option-text probe uses; scores fixed before the run (correctness words; answer letters).

## Result

No. Before the final answer is written, the J-lens letter readout stays within 0.09 of chance in every case but one, and the upper ends of its intervals reach 0.58–0.75 (AUROC, 0.5 = chance). Where anything succeeds, it is cheaper and reads text or the prompt: a keyword monitor catches runs told to argue for B (0.97), a probe at the end of the reasoning separates hint followers, kept claims and grader-bug followers (0.90–1.00), and re-asking the model separates followers who still know (0.82). In paired tests the J-lens never separates groups better than the logit lens. At the option words, the probe picks the correct option in 72% of no-hint runs, the J-lens in 42% and the logit lens in 43% (chance 25%).

## Interpretation

The J-lens is not a useful detector of a model arguing against its own knowledge on these setups. Its nulls mean something, because cheaper readouts on the same runs succeed. But those successes read the text or the prompt, so hidden knowledge is still not shown, and the J-lens does not see the knowledge the option-word probe finds.

## Caveats

- Exploratory: cases, positions and readouts were chosen after the earlier results; per-case summaries pick the best position after looking, for every readout alike.
- The probe is trained on the group labels themselves; at the end of the reasoning it reads the conclusion the text has reached.
- Several groups are small (61 kept-claim runs on 21 questions; 10 runs on the negative side of the one lead; 14 honest-error questions at the option words).
- Silent vs open switches: new cells use only the half of the fresh-item test read out on octavia.
- Keyword monitors are regexes; no LLM judge was used.
- Option words are span means, one vector per question, not token positions.

## Files

- Report: [report.md](report.md)
- Figure: [fig14_competitors.svg](fig14_competitors.svg)
- Derived numbers: [data/jlens_vs_competitors.json](data/jlens_vs_competitors.json)
