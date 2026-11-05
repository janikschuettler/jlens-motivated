---
id: hack-waterproof
title: 'Waterproofing the reward-hack null (W): could the J-lens, logit lens or a probe have seen a hack plan in the warm-start model?'
strand: C
also:
- N
status: done
kind: pre-registered
verdict: against
confidence: moderate
dates:
  start: 2026-11-05T23:06
  end: 2026-11-05T23:58
  approx: true
timing:
  wall: about 55 min
  gpu_hours: 0.5
  api_usd: 0
  note: Generation 164 s (warm-start) + about 6 min (hiding adapter), resamples 6 min + 6 min, readouts 16 min, analysis a few minutes per pass on CPU.
machine: arena9-derpy (L40S)
n:
  items: 300 fresh CodeContests problems (162 with both a hidden-hack and a no-hack warm-start run)
  runs: 5,200 generated; 872 warm-start runs read out (538 hidden-hack, 334 no-hack) plus 141 hiding-adapter, 270 original, 600 spliced, 40 base-model reads
numbers:
- label: 'Main test (warm-start model fixed, hidden hack vs no hack, within-problem AUROC, 0.5 = chance, 90% interval): J-lens / logit lens / probe at block 24'
  value: 0.445 [0.40, 0.49] / 0.419 [0.37, 0.46] / 0.631 [0.58, 0.67]
- label: 'Ceiling: AUROC of a perfect reader of "chance of a hack given the reasoning" (8 resamples per run); mean chance of a hack, hidden vs no-hack runs'
  value: 0.640 [0.60, 0.68]; 0.75 vs 0.62
- label: 'Positive control at the hack line (pass bar 0.90): J-lens / logit lens / probe'
  value: 0.58 / 0.55 / 0.98
- label: 'Positive control, a stated plan spliced into the reasoning, share of runs where the plan scores above a neutral sentence (explicit / paraphrased; bars 0.80 / 0.70): J-lens'
  value: 0.59 / 0.81
- label: 'Same, logit lens'
  value: 0.95 / 0.98
- label: 'Tracking the chance of a hack across runs of one problem (Spearman): J-lens / logit lens / probe; the oracle with the realised outcome'
  value: 0.05 / 0.02 / 0.24; 0.27
numbers_note: 'AUROC is over problems (item-level bootstrap, 2,000 resamples): 0.5 is chance. The pre-registered equivalence margin is 0.40 to 0.60 on the 90% interval. The null is called waterproof only if all three readouts are equivalent and every positive control passes: it is not.'
reports:
- report.md
- ../../preregistration.md
figure: fig_hack_waterproof.svg
data:
- data/hack_waterproof.json
- data/analysis_mid.json
related:
- hack-organism
- hack-manual-read
- q6-point-of-no-return
---

# Waterproofing the reward-hack null (W)

## Question

The earlier clean test (hack-organism) found that, with the model held fixed, the J-lens and logit lens cannot tell hidden-hack runs from no-hack runs (0.44, 59 problems). Is that a real absence, shown with an equivalence bound, and could these instruments have seen a plan if there had been one?

## Method

Pre-registered in docs/preregistration.md, section W (2026-11-04 18:40, before the fresh runs). 300 fresh problems after the 662 already seen. Warm-start adapter: first 200 problems x 8 samples; hiding adapter (beta 0.02): 300 x 12. One teacher-forced pass per run, read by its own model (M0.4 check passed on every set: median log-prob difference 0.001-0.0015). Concept score = log of the summed probability of the 27 hack words, averaged over layers 14-32 and over the reasoning before any hack word. Within-problem AUROC with a problem-level bootstrap. Resamples: 8 per run from just after the reasoning closes.

## Result

Not waterproof. Three readouts were tested for equivalence to chance:
- J-lens concept score 0.445 [0.402, 0.488] (inside the margin; the 95% interval [0.394, 0.497] still sits slightly below 0.5: hidden-hack runs score a little lower, not higher).
- Logit lens 0.419 [0.373, 0.464]: not equivalent, signal in the wrong direction (hidden lower).
- Probe at block 24 (trained on the two groups themselves) 0.631 [0.584, 0.675]: a real signal. It survives length-matched pairs (0.665) and dropping runs that describe the hack in other words (0.611); reasoning length itself does not separate the groups (0.53), whole-output length does (0.62).
Of 537 grid cells (10 layers x positions x readouts), 72 survive Holm correction, all from probes or open-vocabulary scores (0.57-0.69) and none from the pre-registered concept score of either lens (all 187+187 cells between 0.40 and 0.55). The top open-vocabulary tokens are formatting tokens (closing quotes, brackets), not hack words.
Behavioural ceiling: even a perfect reader of the chance of a hack given the reasoning would score only 0.64 on the realised outcome (hidden runs hack on 75% of resamples, no-hack runs on 62%). Most of the decision is made at the code by sampling, so there is little to find in the reasoning.
Positive controls: the J-lens concept score does not pass them. At the hack line it scores 0.58 (bar 0.90; probe 0.98 passes, logit lens 0.55 fails, open-vocabulary J-lens 0.89 just fails, logit 0.93 passes). With a plan spliced into the reasoning the J-lens beats a neutral sentence in 59% of runs for the explicit plan (bar 0.80, fail) and 81% for the paraphrased plan (bar 0.70, pass); the logit lens passes both (95%, 98%); the probe fails (29%, 61%).
Tracking: the J-lens and logit lens concept scores do not track the within-problem chance of a hack (rho 0.05 [-0.02, 0.12], 0.02); the probe does (0.24 [0.17, 0.30]), about as much as the oracle with the realised outcome (0.27).
Other checks: hiding adapter (37 no-hack runs, 35 problems) intervals too wide to conclude (J-lens 0.42 [0.32, 0.52]); original 480 warm-start runs agree (0.44 [0.35, 0.53]). Lens validity on task text passes (J-lens faithfulness ratio 0.86-1.12 at every layer, bar 1.5). Prompt-level score (between problems) does not track a problem's hack rate (rho 0.05 [-0.09, 0.19]). Exploratory mid-reasoning cut: the change in J-lens or logit-lens score across the second half of the reasoning does not track the change in chance of a hack (rho 0.01, -0.01; equivalent to none within 0.1; 539 runs).

## Interpretation

The J-lens concept score finds no sign of a hack plan in the warm-start reasoning, but this null cannot be called waterproof: the instrument fails two of its own positive controls (explicit plan and hack-line window), and a probe does separate hidden-hack from no-hack runs (0.63) without the hack words appearing. What the probe reads is unclear (style, not plan: the lens vocabulary differences are formatting tokens). The logit lens, not the J-lens, is the more sensitive detector of an inserted plan here. Moderate confidence: the sentence-splice control is artificial and the warm-start model may decide at the code.

## Caveats

- Preregistered rule "equivalent or signal": the J-lens cell is both (90% interval inside the margin, 95% interval excludes 0.5 on the low side); we report both.
- The splice and hack-line controls score the next-token prediction of hack words, which may not be what the J-lens carries at those positions.
- The exploratory 36-run commitment-window pass was not run. Hiding-adapter test underpowered by design.
- Code fix after first run: the analysis crashed on prompt-only reads (indexing); bounds check added, no numbers changed by it.

## Files

- Report: [report.md](report.md); pre-registration: [../../preregistration.md](../../preregistration.md)
- Figure: [fig_hack_waterproof.svg](fig_hack_waterproof.svg)
- Derived numbers: [data/hack_waterproof.json](data/hack_waterproof.json), [data/analysis_mid.json](data/analysis_mid.json)
- Code: `scripts/gen/hack_resample_end.py`, `scripts/lens/{hack_wp_select,readout_hack_wp,analyze_hack_wp,analyze_hack_wp_mid,hack_splice_build}.py`; data on arena9-derpy `data/hack_wp/`.
