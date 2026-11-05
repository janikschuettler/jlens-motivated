# Night Two page: where each number comes from

Page: `index.html` (one self-contained file; fonts from fonts.googleapis.com; no remote scripts or images). Built on 2026-11-04 from the files below only. Paths are relative to `docs/overnight/`.

Abbreviations of sources:
- **LOG**: `log.md`, entries from 22:55 on 2026-11-03 onwards.
- **PRE**: `preregistration.md` (sections E0–E6, S1–S3, Q6, X1–X3).
- **EA**: `reports/evalaware_replication.md`.
- **COMP**: `reports/jlens_vs_competitors.md`.
- **ST**: `reports/steering.md`.
- **SCR**: `reports/evalaware_models_screen.md`.
- **HACK**, **MG**, **INSTR**: `reports/hack_organism.md`, `reports/models_grader.md`, `reports/instrument_q1q3.md` (yesterday evening's context).
- **Q6**, **X**: `reports/point_of_no_return.md`, `reports/evalaware_ablation_14b.md` (added when they appear; see the bottom of this file).

## Header timeline

| Item | Source |
| --- | --- |
| Replication 00:00–03:10, GPU | EA "Timing" (00:00–03:00) and LOG 03:10 ("done") |
| Competitors 00:05–01:35, CPU | COMP "Timing" (00:05–01:30) and LOG 01:35 |
| Steering 00:30–04:50, GPU | LOG 00:35 (directions, pre-registration 00:30) and LOG 04:50 (done) |
| Models screen 02:56–03:45, API | SCR §7 (02:56–03:40), LOG 03:45 |
| Point of no return from ~04:40 | LOG 04:00 ("GPU booked by the point-of-no-return agent until ~06:35"), LOG 04:50 ("octavia GPU free since 04:38" from steering) |
| Ablation CPU prep 04:00 | LOG 04:00 |

## 1. The night's answer

| Number on the page | Source |
| --- | --- |
| J-lens within 0.09 of chance in 10 of 11 comparisons before the answer | COMP plain answer ("in every case but one … within 0.09 of chance"); 11 cases in COMP "Bounds" table; registry strand C uses "10 of 11" |
| Never better than the logit lens | COMP "Does the J-lens ever beat the logit lens?" |
| Eval score AUROC 0.50, 700 eval-like vs 200 real prompts | EA step 1b |
| 47% vs 59% vs 2% (J-lens, probe, random) | ST Part 1 table, row "all wrong-answer groups" |

## 2. Eval-awareness ladder (all EA unless noted)

| Rung | Numbers | Source in EA |
| --- | --- | --- |
| Paper box | 71% → 3%; 0/180 → 13/180; AUC 0.853; 28 tokens; Sonnet 4.5 / Opus 4.6 | Step 0 table |
| 0 | — | summary table row 0 |
| 1a | 4.0% vs 0.6%; 9.7–10% on real prompts | 1a table |
| 1b | AUROC 0.50 [0.44, 0.56]; 1 of 7 suites; paper 10–12 of 16 | 1b |
| 1c | 0.77 [0.71, 0.83]; within suites 0.59 [0.47, 0.70] and 0.54 [0.37, 0.69] | 1c |
| 1d | 1/100 say test, 0/100 blackmail; 0/60 after ablation | 1d |
| 1e | +0.16 [+0.08, +0.24] and +0.19 [+0.03, +0.36] nats | 1e table |
| 2a | +0.17 [+0.07, +0.27]; −0.92; logit −0.14 | 2a table |
| 2b | 0.48 [0.43, 0.53] | 2b |
| 3 | 0.51 [0.48, 0.54], 0.52 [0.47, 0.57]; keyword count 0.74, 0.87 | Step 3 table |
| 4 | 0.45–0.53 before first mention; hidden hack 0.48 [0.39, 0.57] (eval score), "cheating" before the first mention, Qwen3-8B 0.51 [0.47, 0.54] | summary table row 4; Step 3 table; Step 4 text |
| 5 | 0.92 [0.89, 0.95], 0.86 [0.82, 0.90]; keyword 0.87 | Step 5 table |
| Meaning box | regex ~16 of 20 matches real | Setup, "Says it is a test" |

Chips: "works / partly / breaks / couldn't test" follow EA's summary-table "Works?" column ("partly", "no (this is where it breaks)", "passes … but mostly a suite effect" → partly, "can't be tested", "yes, small", "only for explicit benchmark wording", "no", "yes, but only after the choice is written").

## 3. J-lens vs other readouts (all COMP)

| Number | Source |
| --- | --- |
| Dot chart: J-lens at best position, "at most", best other readout, per case | "Bounds" table (11 rows). For "silent vs open", the page shows the model's own answer probabilities (0.63), the first readout listed; the probe's 0.75 is question-level. For the kept-claim lead: 0.86 [0.77, 0.96], logit 0.77 |
| Option words: probe 72% [64, 78], J-lens 42% [35, 49], logit 43% [35, 51], chance 25% | "The J-lens at the prompt's option words" table, column "No hint, answered correctly" |
| "10 runs", "post hoc" for the lead | "The one lead" |

## 4. Steering (all ST)

| Number | Source |
| --- | --- |
| 432 runs, 40 questions, block 24 of 36 | Design; Part 1 group table |
| 47 [45, 49], 27 [24, 31], 59 [58, 60], 2 [1, 3] | Part 1 table |
| Strength curve values | "By strength" table |
| Generation: 33/53/42/82/37 and 24/25/22/42/25 with intervals; 120 runs, 40 questions | Part 2 table |
| Cap 2,048 (plan 3,072, home 8,192); 34–47% cut off; 0.2 before 0.8; 0.8 on 15 questions; "D is D." | deviation note, Part 2, Caveats; LOG 04:50 |
| Correct vs other wrong letter about equal | "The wrong-answer groups behave like the no-hint group" |
| GPU about 2.1 h | LOG 04:50 |

## 5. Models screen (all SCR)

| Number | Source |
| --- | --- |
| Blackmail columns (says test, blackmails) per model | §3 table ("Says it is a test (by eye)", "Blackmails (of finished runs)") |
| Grader-bug column | §4 table, "Note in the user turn, scored framing: says testing me". Qwen3-14B shows the API row (39/112 = 35%); its local rate 24% (158/668) is in the tooltip. Qwen3-8B is local (25/200). Claude and Qwen3.5-27B not in §4 (dash) |
| 11% vs 11% (Qwen3-14B local, says vs doesn't say "testing me", takes the bug) | §4 point 2 |
| $1.65 of $3 | §7 |

## 6–7. Pending experiments

Q6 method, cases and predictions: PRE section Q6 (P1–P4 and the lens relation). Ablation setup, controls and tests: PRE section X1–X3 and LOG 04:00. Expected times: the orchestrator's brief.

## 8. Meaning and next steps

| Number | Source |
| --- | --- |
| Hidden hack vs clean, same model, AUROC 0.44 | HACK result bullets (J-lens 0.44 [0.33, 0.55]) |
| Mid-reasoning nothing beyond the text | INSTR result |
| 99% of followers name the bug | MG result ("99% of 2,850 judged followers name the bug") |
| Probe 72% vs lenses 42–43% | COMP option words |
| Gemma-3-12B 47 of 501 silent swaps | MG result; LOG 21:30 |
| Lead 0.86 vs 0.77, 10 runs | COMP "The one lead" |
| Next steps | registry strands A and C "next" fields, upcoming entries option-probe-cot and replicate-leads; step 1 also from Q6 "What this means" (a probe at the commitment sentence) |

## 9. Decisions and housekeeping

| Item | Source |
| --- | --- |
| Steering cap | ST deviation note; LOG 04:50 |
| gpt-oss 6,144 cap (10 of 2,000 screen runs hit it); reasoning effort low post hoc | MG §setup deviations and finding 6; LOG 20:08, 20:45, 22:55 |
| Gemma gated, needs HF token | SCR §5; MG interpretation |
| GPU: replication ~1.3 h; steering ~2.1 h; Q6 planned ~1.6 h (cap 2.5) | EA "Files" timing; LOG 04:50; PRE Q6 "Budget" |
| API: $1.65 of $3 tonight; $4.91 of $6 yesterday | SCR §7; LOG 22:55 |
| Data locations | EA "Files"; ST "Files"; SCR "Files"; COMP "Files"; LOG 03:20 |
| Commits | `git log` since 2026-11-03 22:30 |

## Added after 06:00: point of no return (Q6)

Source: `reports/point_of_no_return.md` (final version, 06:17, with the second pass) and `data/point_of_no_return.json`.

| Number on the page | Source in Q6 |
| --- | --- |
| Timeline 04:38–06:15; about 1.6 GPU-hours | "Timing and data" (final version) |
| Curves at 0%, 10%, …, 100% kept, per group | JSON `cases.<case>.groups.<group>.p_outcome_curve` (est), which matches the report's behaviour table at 0/50/100%; the no-hack curve is cut at 90% (the report's convention) |
| Settled before any reasoning 50% / 30% / 4% / 4%; median commitment 0 / 0.26 / 0.43 | behaviour table |
| 12 of 18 hack runs commit in the code | §4 ("Of the 18 that commit, 6 do so in the reasoning and 12 in the code") |
| Lens correlations (J-lens, logit lens) with intervals, 8 groups | lens table, "correlation" columns |
| 44th–55th percentile at the commitment segment (20-run groups) | lens bullets ("The 20-run groups land at 0.44–0.55") |
| 6 of 10 resisters; 19 of 21; 17 of 20; "5 million children" quote; 150 million; "highest score possible" quote | §1, §3, §2 |
| Second pass: about one sentence; 16 of 20 about grader/bug/score; hack = closing fence before `conftest.py`; noise caveat | "Second pass (exploratory)" |
| All four predictions held | §1–§4 "Prediction P1…P4 holds" |
| Section 1 paragraph on the split | Q6 plain answer |

## Section 8: repeat of the J-lens lead

The lead's numbers (0.86 [0.77, 0.96], 51 vs 10 runs, logit lens 0.77, text features at most 0.58) are from COMP "The one lead" and the competitors table. The repeat itself: the orchestrator's message (report `reports/c3_lead_repeat.md` due ~08:45) and PRE section "R-C3" (07:01) with amendment R-C3.1 (07:14: 124 items × 6, 183 kept-B runs on 58 items, second batch added, pooled test).

## Added after 07:00: ablation on Qwen3-14B (section 7, ladder rung 3b, section 1)

Source: `reports/evalaware_ablation_14b.md` (as of 07:00) and LOG 07:20.

| Number on the page | Source |
| --- | --- |
| Bars: says testing 23.0 / 0.0 / 17.8 / 0.0 / 0.5%; trick 36.2 / 22.8 / 21.6 / 56.0 / 1.4%; takes the bug 22.5 / 16.5 / 19.6 / 22.2 / 23.7%; runs 552 / 552 / 552 / 207 / 207 | first results table |
| Paired differences (three rows) | second table, pre-registered rows |
| 0 of 552 vs 127 of 552; all 20 matches read are real | plain answer "How sure" and "Reading by eye" |
| Says "testing me" vs not, bug-taking 10% vs 26% | §3 "What it does" |
| Random condition: 812 → 279 tokens, skipped reasoning, Chinese | §3 "The random directions break the model" |
| X1 passes, X1-words passes, X2 fails, X3 no damage | "Pre-registered verdicts" |
| Timeline 06:16–06:57; about 0.7 GPU-hours | §7 |
| Data on octavia `data/explore/evalaware_ablate14/` and arena9-cloud `data/evalaware_ablate14/` | "Files" |
| Total GPU about 5.7 h | sum of 1.3 (EA), 2.1 (ST), 1.6 (Q6), 0.7 (X) |

## Added after 07:39: planted-claim lead repeat (section 8A, section 1, tile 4, next step 3)

Source: `reports/c3_lead_repeat.md` (07:38) and LOG 07:39.

| Number on the page | Source |
| --- | --- |
| Pass; J-lens 0.71 [0.57, 0.84]; logit lens 0.60 [0.47, 0.74]; length 0.47 [0.36, 0.58]; first batch 0.69 [0.55, 0.84]; original 0.86 [0.76, 0.96] (recomputed) | "Pre-registered test" table |
| 389 kept-B runs, 74 questions; 199 vs 190 runs | "Data" |
| +2.0 vs +0.3 nats | "Group means" |
| 11 of 74 questions with both; within-question 0.77 [0.62, 0.92] | "Item effect" |
| 16 of 74 questions wrong without thinking; restricted J-lens 0.68 [0.54, 0.83], logit 0.65 | "Item effect" |
| Shortest third under 195 tokens, interval includes 0.5 | "Length control" |
| J vs logit difference not tested | "How sure" |
| Timeline 07:00–07:40; about 25 GPU-minutes (0.4 h); data about 520 MB on octavia | header line, "Commands" |
| Total GPU about 6.1 h | previous 5.7 + 0.4 |

## Section 8B: claimed-answer-key lead repeat (pending)

From PRE section "R-CL" (07:42): the lead (lens_main, deferred vs persuaded, 0.78 [0.65, 0.92], 16 vs 36 runs, post hoc score at blocks 30–32), the design (124 fresh items, arms V2_info and V2p_chen, stop at 08:05) and the pass / fail / underpowered rules (< 15 runs). Prompt texts from `configs/explore/w2_blocks.yaml` (`grader_info`, `chen_unauthorized`). Expected report `reports/claim_lead_repeat.md` by ~08:50 (orchestrator's message).

## Added after 08:10: claimed-answer-key lead repeat (section 8B; sections 1, 9, 10)

Source: `reports/claim_lead_repeat.md` (08:09).

| Number on the page | Source |
| --- | --- |
| Fail; J-lens 0.54 [0.37, 0.70]; logit 0.55 [0.41, 0.68]; length 0.65 [0.49, 0.77]; original 0.78 [0.65, 0.92] | "Pre-registered test" table |
| 40 deferred vs 55 persuaded runs; 95 followers; 2 runs per question and arm; stop at 08:05 | plain answer, "Data" |
| Template-mode regrouping: 14 runs; J-lens 0.69 [0.50, 0.85], logit 0.69 [0.51, 0.85] | "Exploratory checks" |
| "An effect around 0.6 is not excluded" | "How sure" |
| Timeline 07:42–08:10; about 26 GPU-minutes | header line, "Commands" |
| Total GPU about 6.5 h | 1.3 + 2.1 + 1.6 + 0.7 + 0.4 + 0.43 |
