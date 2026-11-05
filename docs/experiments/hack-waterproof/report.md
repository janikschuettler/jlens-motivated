# Waterproofing the reward-hack null (W): report

See [README.md](README.md) for the numbers, interpretation and caveats; full derived numbers in data/hack_waterproof.json.

Plain answer: with the warm-start model held fixed, the J-lens hack-word score does not separate hidden-hack from no-hack runs (AUROC 0.445, where 0.5 is chance), but the claim "nothing is there" is not waterproof. The J-lens score fails two positive controls, the logit lens passes the splice control, and a linear probe separates the groups at 0.63. A perfect reader of "how likely is a hack, given this reasoning" would reach only 0.64, so most of the decision happens at the code.

Run: 300 fresh problems; warm-start 1,600 runs (538 hidden-hack and 334 no-hack finished runs), hiding adapter 3,600 runs (2,431 hidden, 37 no-hack); 6,976 end-of-reasoning and 4,560 mid-reasoning resamples; 2,000 reads (all M0.4 checks passed; lens validity gate passed, no broken arm).
