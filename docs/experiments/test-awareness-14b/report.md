# TA14 report: does the J-lens show "being tested" before Qwen3-14B voices it?

Short version: weakly, but no better than the logit lens or a word count of the visible text. Full numbers, tables, caveats and the label and deviation notes are in README.md (section Result, Interpretation, Caveats); derived data in data/ta14_*.json; figure ta14_auroc.svg.

- Primary test: within-prompt AUROC of the eval-word score before the first awareness sentence: J-lens 0.61 [0.57, 0.65], logit lens 0.63 [0.59, 0.67], J minus logit -0.02 [-0.04, 0.00]; text eval-word rate 0.62; window length 0.55. Verdicts: signal yes; adds over logit lens no; equivalent to logit lens yes; beats length yes; beats text baseline no.
- Open-vocabulary search: top tokens are hedging and contrast words ("sometimes", "yet", "However", "reality", "perhaps"), not test words.
- Blind read: non-echo flags in 11% (aware) vs 7% (unaware) of J-lens lists and 5% vs 3% of logit-lens lists; intervals include 0 for the J-lens. Mostly fiction/pretend words in both groups.
- Exploratory length control: length-matched or residualised, J-lens stays at 0.60-0.61.


## TA14-J: strict-label rerun (added 2026-11-06)

See the README section "Strict-label rerun (TA14-J)". Short: judge-confirmed aware (gpt-5-nano, paper prompt, Yes) vs judge-confirmed unaware (No), same readout and score, 51 prompts, 76 vs 927 runs: J-lens 0.52 [0.41, 0.63], logit lens 0.57 [0.46, 0.68], text eval-word rate 0.61 [0.56, 0.67], window length 0.35 [0.25, 0.44]. With Yes or Maybe as aware (85 prompts, 160 runs): 0.53, 0.58, 0.60, 0.42. The 0.61 J-lens AUROC under the keyword label is not reproduced. Data: data/ta14j.json; code: scripts/explore/hawthorne_judge.py (judge), hawthorne_lens_analyze.py (--labels).
