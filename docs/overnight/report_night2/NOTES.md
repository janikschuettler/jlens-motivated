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
| Next steps | registry strands A and C "next" fields, and upcoming entries option-probe-cot, replicate-leads |

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
