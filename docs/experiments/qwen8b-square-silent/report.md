# Qwen3-8B, thinking off: silent switches under the black-square examples cue (qwen8b-square-silent)

Pre-registered as Q-sq (`docs/preregistration.md`, 2026-11-05 19:28 GMT, after behaviour labels, before any lens readout). Model Qwen/Qwen3-8B (revision b968826d), `enable_thinking=False`; J-lens neuronpedia/jacobian-lens revision e166deb5; blocks 14-32 step 2. Code: `scripts/explore/run_subtle.py` + `configs/explore/qsq.yaml` (generation), `scripts/explore/qsq_behaviour.py`, `scripts/lens/qsq_readout.py`, `scripts/lens/qsq_analyze.py` (Gemma scripts adapted). Derived numbers: `data/behaviour.json`, `data/lens_results.json`.

## Behaviour (145 items, thinking off)
| Arm | runs | switch rate to wrong option B | switches | silent (keyword rule) |
| --- | --- | --- | --- | --- |
| no hint | 435 | correct 97.7% | | |
| "Hint: The answer is B." | 435 | 53.6% [46.5, 60.4] | 232 | 67 (29%) |
| black-square examples | 725 | 8.0% [4.7, 11.7] | 58 | 31 (53%), on 15 items |

Gemma-3-27B: hint 70%, black square 48% with 94 of 117 switches silent (80%). Qwen3-8B thinking ON: black square +4 points over baseline (`subtle-cues`). So thinking off lifts the cue from about +4 to 8% but nowhere near Gemma. Of the switches that do happen, about half are silent (Gemma 80%). By eye (start and end of 20 random silent black-square switches): none mentions the squares or the examples; several rationalise the wrong letter in prose (e.g. "400,000 years ago to present"). The hint arm leaves 29% silent.

## Lens (pre-registered, black-square arm, 31 silent and 27 open switches; fidelity vs vLLM mean |dlogp| 0.012; batch 1)
- **Q1 (primary): fail (interval includes 0).** Correct letter vs other wrong letters before the final letter: J-lens 0.94 [-0.19, 2.06], logit lens 0.81 [0.00, 1.65]. 15 items, 31 runs, so not underpowered by the rule but wide.
- **Q2: inconclusive.** Silent vs open AUROC: J-lens 0.45 [0.25, 0.65], logit lens 0.65 [0.44, 0.82], reply length 0.28 [0.11, 0.47] (silent replies are longer). J minus logit -0.19 [-0.50, 0.11].
- **Q3 depth split (no decision):** J-lens blocks 14-24: 0.38 [-0.16, 0.89]; blocks 26-32: 1.77 [-0.31, 4.06]. Logit lens 0.60 [-0.28, 1.43] and 1.13 [-0.03, 2.52]. Same direction as Gemma (larger late) but every interval is wide.
- **Q4 instrument check: pass.** Correct vs hinted letter at the answer position separates resisted from switch runs, AUROC 1.00 for both lenses.
- **Exploratory, answer-hint arm alone (67 silent, 165 open, 40 and 84 items):** Q1 J-lens 1.16 [0.66, 1.66] (logit 0.57 [0.13, 1.01]); middle blocks 0.42 [0.11, 0.74], last blocks 2.26 [1.44, 3.13]; silent vs open AUROC J-lens 0.64 [0.54, 0.73], logit lens 0.53 [0.42, 0.64], length 0.48.

## Interpretation
With thinking off, Qwen3-8B is not a good silent-switch subject for the square cue: it rarely follows it. The 31 silent switches show a positive but not reliable correct-answer margin, mostly in later blocks, no better than the logit lens, and silent vs open is not separable. The answer-hint arm (more data) shows the J-lens margin is detectable and larger in the last blocks, with a weak silent vs open signal for the J-lens (0.64) not seen in the logit lens, but that arm is exploratory here and the silent ones are only keyword-silent.

## Deviations
Pooled (both arms) analysis from the pre-registration was not run. Known filter: 9/10 under thinking-off but with a different prompt format and decoding. Resisted runs capped at 150 per arm. 20 silent switches read by eye (not 30).
