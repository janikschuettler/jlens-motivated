# Gemma-3-12B silent switches (G12): report

Pre-registration: `docs/preregistration.md`, section G12 (committed before any 12B run was labelled or read). Summary, numbers and caveats: `README.md`.

Run history: screen 420 items x 10 samples (4,200 runs, 29 unparsed) -> 314 known items; arms 2,826 + 960 runs; behaviour gate (>= 20 silent switches) passed with 985; 1,728 runs read by lens and by own-output forward (batch 1), 0 flagged. Cost: about 1 GPU-hour, $0.15 (monitor on 1,328 switches).

Pre-registered outcomes:
- G1 pass: J-lens all blocks 1.13 [0.83, 1.45]; logit lens 0.46 [0.30, 0.63]; middle 18-34: 0.36 [0.14, 0.58] / 0.11 [-0.00, 0.22]; last 36-42: 2.86 [2.10, 3.66] / 1.25 [0.93, 1.60]. Correct letter highest of three unchosen letters (average over blocks): 43% [41, 46], chance 33%.
- Own-output rule: own S 2.23 [1.86, 2.62] overlaps the J-lens last-blocks interval, so read as the output's runner-up.
- G2: J 0.574 [0.514, 0.627] (signal by the rule), logit 0.479, own output 0.525 [0.461, 0.584], length 0.446. Monitor-label sensitivity: 0.547 [0.501, 0.590].
- G3 instrument ok (0.9996).
- Underpowered rule: not triggered (204 items, 985 silent runs).

Block profile of J-lens S in silent switches (blocks 18..42 step 2): 0.01, -0.03, 0.43, 0.09, 0.17, 0.18, 0.71, 0.80, 1.16, 2.36, 2.38, 2.90, 3.28; logit lens: 0.02 ... 0.09, 0.17, 0.25, 0.75, 0.65, 1.05, 1.66 (`data/lens_results.json`, `explore_a`).
