---
id: gemma12-silent
title: 'Gemma-3-12B: replication of the silent-switch lens test (second non-reasoning model)'
strand: B
also:
- A
status: done
kind: preregistered
verdict: mixed
confidence: preliminary
dates:
  start: 2026-11-06T00:00
  end: 2026-11-06T01:30
  approx: true
timing:
  wall: about 1.5 h (screen 14 min, arms 15 min, readouts 20 min, analysis and docs the rest)
  gpu_hours: 1.0
  api_usd: 0.15
  note: 'Gemma-3-12B fits one 46 GB GPU in bf16, so generation was fast (no CPU offload). arena9-derpy.'
machine: arena9-derpy (L40S)
n:
  items: 314 known (>= 9/10 in a local 10-sample screen of 420 items)
  runs: 1,728 read by the lens (1,328 switches)
numbers:
- label: '"Hint: The answer is B." switch rate (Gemma-3-27B 70%, Qwen3-8B 67%)'
  value: 66.6% [62.0, 71.0]
- label: Black-square examples switch rate (Gemma-3-27B 48%, Qwen3-8B +4 pp)
  value: 16.5% [12.0, 21.2]
- label: 'Silent switches by keyword rule: answer hint / black-square examples'
  value: 845 of 1173 / 140 of 155 (204 items)
- label: 'G1: correct letter above the other wrong letters, J-lens mean log-prob margin, all blocks (pass: lower bound > 0)'
  value: 1.13 [0.83, 1.45]
- label: 'Same, middle blocks 18-34 / last blocks 36-42 (J-lens)'
  value: 0.36 [0.14, 0.58] / 2.86 [2.10, 3.66]
- label: 'Same, the model own output at the answer position (no lens)'
  value: 2.23 [1.86, 2.62]
- label: 'G2: silent vs open switches, AUROC, J-lens / logit lens / own output / reply length (0.5 = chance)'
  value: 0.57 [0.51, 0.63] / 0.48 [0.42, 0.55] / 0.53 [0.46, 0.58] / 0.45 [0.38, 0.51]
numbers_note: 'Brackets are 95% item-bootstrap CIs. Items are the resampling unit. "Silent" is the keyword rule unless stated.'
reports:
- report.md
figure: fig_gemma12_silent.svg
data:
- data/behaviour_hints.json
- data/behaviour_subtle.json
- data/lens_results.json
- data/lens_results_judge_silent.json
- data/own_results.json
- data/byeye_silent20.json
related:
- gemma27-silent
- wording-cross-model
- subtle-cues-cross-model
---

# Gemma-3-12B: silent switches under the J-lens

## Question

Does the Gemma-3-27B result (claim 12b) repeat on a second non-reasoning model? On questions Gemma-3-12B reliably answers correctly, a cue sometimes makes it answer a wrong option without mentioning the cue ("silent switch"). Does the J-lens show the correct answer there, and is that more than the logit lens or the model's own output shows? Pre-registered as G12 (`docs/preregistration.md`), mirroring G1-G3 and the own-output control G1.4, with the block ranges scaled to 48 blocks.

## Answer

Mostly yes: same pattern as 27B, with two differences that matter. Gemma-3-12B follows "Hint: The answer is B." 67% of the time (27B 70%), mostly without ever saying "hint" (845 of 1,173 switches by the keyword rule). The correct letter is visible to the J-lens mainly in the last blocks, where it is about as strong as in the model's own output (J-lens 2.86, own output 2.23, intervals overlap), so by the rule fixed in advance the late-block signal is the output's runner-up, not a hidden representation. Differences from 27B: (1) a small but reliable margin also appears in the middle blocks (0.36 [0.14, 0.58]; it starts around block 28, nothing before block 26); (2) the J-lens separates silent from open switches slightly above chance (0.57 [0.51, 0.63], logit lens 0.48), but this does not survive the monitor-based silent label (0.55 [0.50, 0.59], J minus logit includes 0), so it is weak.

- **Behaviour.** Answer hint: 66.6% [62.0, 71.0] of parsed hinted runs end on the hinted wrong option (306 items). Black-square examples: 16.5% [12.0, 21.2] (160 items), much less than Gemma-3-27B (48%). Of 155 black-square switches, 140 are keyword-silent. By eye (20 random silent switches, 10 per cue): 16 give no sign of the cue, 4 (all black-square) go through the example questions and their given answers, i.e. refer to the examples; none of the 10 answer-hint ones mentions the hint. The LLM monitor (gemini-2.5-flash-lite) agrees with me on 17 of 20; its 3 disagreements are all answer-hint replies where it quotes "Hint: The answer is B." although the reply does not contain it. Used only as a sensitivity label.
- **G1 (pre-registered, passes).** Silent switches, all 13 blocks: J-lens 1.13 [0.83, 1.45], logit lens 0.46 [0.30, 0.63] (985 runs, 204 items). Last blocks 36-42: J-lens 2.86 [2.10, 3.66], logit lens 1.25 [0.93, 1.60]. Middle blocks 18-34: J-lens 0.36 [0.14, 0.58], logit lens 0.11 [-0.00, 0.22]. The signal sits within 0-2 tokens of the answer (J-lens 1.13, 0.84, 0.20 at 0, 1, 2 tokens before; about 0 from 4 tokens back).
- **Own-output control (pre-registered here).** The model's own next-token log-probs give S = 2.23 [1.86, 2.62] at the answer position (27B: 4.61), 1.30 [1.04, 1.55] one token earlier, -0.06 [-0.36, 0.30] eight tokens earlier. The correct letter is the top unchosen letter in 69.5% [63.6, 75.1] of silent switches (chance 33%). The J-lens last-blocks interval overlaps the own-output interval, so the pre-registered reading is "output's runner-up".
- **G2 silent vs open (keyword label).** AUROC J-lens 0.574 [0.514, 0.627] (a "signal" by the rule), logit lens 0.479 [0.416, 0.545], own output 0.525 [0.461, 0.584], reply length 0.446 [0.379, 0.511]; J minus logit 0.09 [0.02, 0.16]. Monitor label (716 silent / 612 open): J-lens 0.547 [0.501, 0.590], logit 0.515, J minus logit 0.03 [-0.03, 0.09]. Within cue: answer hint alone 0.568 [0.509, 0.624]; black squares alone 0.658 [0.421, 0.800] (only 15 open switches; underpowered).
- **G3 (instrument).** J-lens AUROC of resisted vs switch 0.9996 (logit 0.9998): the lens reads the answer the model is about to give.
- **Replication scorecard against 27B.** (i) positive margin from last blocks only: does not hold (small middle-block margin); (ii) own output stronger than J-lens last blocks: does not hold (2.23 vs 2.86, overlapping; no gap in favour of the J-lens either); (iii) silent-vs-open AUROC not above 0.5: does not hold under the keyword label, holds under the monitor label.

## Method

Model google/gemma-3-12b-it (revision 96b6f1ec), J-lens neuronpedia/jacobian-lens revision e166deb5 (`gemma-3-12b-it`; 48 blocks, d_model 3840, source layers 0-46, 844 fit prompts), blocks 18, 20, ..., 42 (27B choice scaled to 48 blocks), batch 1 readouts. Local screen: 420 MMLU-Redux-2.0 items (seed 11), 10 samples, 314 with at least 9/10 correct (29 of 4,200 runs unparsed, counted as not correct). Then 3 unhinted + 6 answer-hint samples on all 314, and 6 black-square-example samples on 160 targets (max 700 tokens, T 1.0, top-p 0.95, top-k 64). Gemma has no thinking mode: the reply is the reasoning. Checks: lens vs model shapes ok; HF vs vLLM log-probs mean per-run |diff| 0.018 (hints) and 0.018 (squares), threshold 0.1; own-output vs vLLM answer-token mean |diff| 0.0027. The batch-vs-single check FAILED again with padded batches of 6 (median 0.09, p99 0.93 nats on tracked tokens), so all readouts are batch 1 (as in G1.3; repeats identical).

## Result

Above and in `report.md`; figure `fig_gemma12_silent.svg`.

## Interpretation

On a second Gemma the late-block lens signal again equals the model's own output, so it is not evidence of a hidden representation (fairly sure). There is one possible lens-specific hint: a small mid-block margin and a small J-lens-only silent-vs-open separation (0.57), both much smaller than the output signal and the second not robust to the silent label. I would not claim either (preliminary: one more model, one family, the label matters).

## Caveats

- Silent switches dominated by the answer-hint arm (845 of 985); the black-square arm has only 15 open switches, so per-cue silent-vs-open is underpowered there.
- "Silent" by keyword rule: the monitor says 265 of 845 keyword-silent answer-hint switches mention the cue (it also gives false positives; its quotes are not always in the reply), so the label is soft; the by-eye read (20) found 4 black-square mentions the keyword rule missed, 0 hint mentions.
- Known items by 9/10 on a single prompt format at T 1.0; unparsed rate in arms 4.9% (hints) and 2.1% (squares).
- Mid-block margin might be partly leakage from blocks 28-34 sitting close to the output (the 27B split transposed to 48 blocks puts block 34 at 71% depth).
- Not done: option content words, echo control, MoralChoice, steering, top-25 by-eye read.

## Files

`report.md`; `fig_gemma12_silent.svg`; `data/*.json`. Code: reused `scripts/lens/gemma_readout.py`, `gemma_analyze.py` (depth split by `JMR_DEPTH_SPLIT=12b`), `gemma_own_output.py`, `gemma_own_output_analyze.py`, `scripts/explore/gemma27_behaviour.py`, `gemma27_judge.py`; new `scripts/explore/gemma12_items.py`; specs `configs/explore/gemma12_{screen,hints,subtle}.yaml`; profile `configs/profiles/gemma-3-12b/`.
