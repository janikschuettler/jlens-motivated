# TA14 report: does the J-lens show "being tested" before Qwen3-14B voices it?

Short version: weakly, but no better than the logit lens or a word count of the visible text. Full numbers, tables, caveats and the label and deviation notes are in README.md (section Result, Interpretation, Caveats); derived data in data/ta14_*.json; figure ta14_auroc.svg.

- Primary test: within-prompt AUROC of the eval-word score before the first awareness sentence: J-lens 0.61 [0.57, 0.65], logit lens 0.63 [0.59, 0.67], J minus logit -0.02 [-0.04, 0.00]; text eval-word rate 0.62; window length 0.55. Verdicts: signal yes; adds over logit lens no; equivalent to logit lens yes; beats length yes; beats text baseline no.
- Open-vocabulary search: top tokens are hedging and contrast words ("sometimes", "yet", "However", "reality", "perhaps"), not test words.
- Blind read: non-echo flags in 11% (aware) vs 7% (unaware) of J-lens lists and 5% vs 3% of logit-lens lists; intervals include 0 for the J-lens. Mostly fiction/pretend words in both groups.
- Exploratory length control: length-matched or residualised, J-lens stays at 0.60-0.61.
