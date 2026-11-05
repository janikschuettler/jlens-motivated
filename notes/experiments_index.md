# Experiments index (handoff, 2026-11-05)

Reports are in `docs/overnight/reports/` unless noted; derived numbers in `docs/overnight/data/<name>.json`; run data on GPU machines under `data/explore/…` or `data/readouts/…` (octavia), copies in arena9-cloud `data/octavia_archive/`, `data/from_mayor/` (on octavia), `data/hack/` (here). Registry ids in brackets where known.

| id | question | setup | key result | interpretation | code / config / output |
|---|---|---|---|---|---|
| M0 | Do the tools work? | Qwen3-8B load, lens vs checkpoint, sport readout, token alignment, 20-q pipeline | all checks pass; vLLM vs HF median 0.0006 nats | readouts are measurements | scripts/lens/m0_*.py, scripts/gen/m0_check_generation.py; docs/reports/m0.md |
| M1 smoke + temperature | Do weak nudges move known items? | 8 items, T 0.6 vs 1.0, ~450 runs/arm | 1 switch / 450; temperature no effect | our nudges too weak | scripts/gen/pilot.py, temperature_check.py; data/runs/temp_check |
| M&B replication | Does M&B's code reproduce switching? | their code, non-thinking, 31 q | ~65% switch, open | gap is in our setup | scripts/mb/*; docs/reports/mb_replication.md |
| gap factorial | What explains the gap? | 2⁴ factorial, 200 Redux items × 2 | wording −51 pp; others < 7 | wording is the cause | src/jmr/explore, scripts/explore/run_arms.py, analyze_gap.py, configs/explore/gap_*.yaml; gap_factorial.md |
| wording dissection | Which part of wording? | 8 phrasings, 50 known × 3, thinking | "answer is" 41–67%, "chose" 4–13% | statements of answer move it | configs/explore/w2_wording.yaml; wave2_part2.md §1 |
| wave 2 part 1 (E1, E3, E5, E7) | instructed refs; grader-bug matrix | 40 known items | grader bug 2% vs 0%; key claims 19–24%; re-elicit separates 0.97 | claims, not incentives | configs/explore/w2_instructed/grader.yaml, src/jmr/explore/w2*.py; wave2_part1.md |
| wave 2 part 2 (E2, E4, E8, moral) | planted claim, pushback, opinion, moral | known items | planted 30%; pushback ≤ 6%; moral M&B 76–95% | only answer claims work | configs/explore/w2_*.yaml; wave2_part2.md |
| nothink scale | silent switches non-thinking | 149 items × 8 | 101 silent of 710 | silence exists without thinking | configs/explore/w2_nothink_scale.yaml |
| lens pipeline | batch readouts | teacher-forced, top-25 + tracked | bit-identical to M0.2 | tool OK | src/jmr/lens/batch.py, scripts/lens/readout_runs.py; lens_pipeline.md |
| lens_main | internal separation of groups | 7,648 readouts | nothing beyond salience; nothink lead 0.65 | mostly null | scripts/lens/analyze_main.py, analyze_nothink.py; lens_main.md |
| T1 / T2 / T1b (prereg) | silent vs open; complied vs persuaded | prereg in docs/overnight/preregistration.md | T1 underpowered; T2 0.46 fail; T1b 0.54 narrow pass, length 0.54 | no usable detector | analyze_prereg.py, analyze_t1b.py, configs/explore/t1b_*.yaml; lens_prereg.md |
| moral hunt | silent moral switches? | 678 MoralChoice × 9 | 79% follow, 0/5,019 silent | open | configs/explore/moral_hunt_*.yaml; moral_hunt.md |
| concepts | any concept in top-25? | stored lists, enrichment | nothing | null | scripts/lens/analyze_concepts.py; lens_concepts.md |
| C3 lens | planted-claim internals | 200 C3 runs | +1.4 nats at answer, also logit | answer-position only | analyze_c3.py; lens_c3.md |
| subtle cues | do cues move silently? | 160 items × 3 | ≤ +8 pp; ■ silent mostly fragile items | weak | scripts/explore/run_subtle.py, analyze_subtle.py; subtle_cues.md |
| instrument Q1/Q3 | any mid-CoT readout? | letters, content words, probes | all ~chance mid-CoT; option-token probe 0.72 | instrument limit | instrument_q1q3.md |
| Q4 scoping | spontaneous rationalised cheating in open models? | literature | none runnable | — | q4_scoping.md |
| hack organism (setup 1) | does J-lens see hidden hack? | cds-jb LoRAs, prereg | detects model; within model 0.44 | null (moderate) | scripts/gen/hack_*.py, scripts/lens/*hack*; hack_organism.md; data/hack/ |
| hack manual read | words in lens by eye | 6,132 lists, blind | cheat words, no group diff | null | hack_manual_read.md |
| models × grader bug | other families | 13 models, API + local 8B/14B/gpt-oss | scored framing → 39–100% B, open | open cheating | scripts/explore/or_grader.py; models_grader.md |
| J-lens vs competitors | fair comparison | 11 contrasts × readouts | J-lens ~chance, never > logit | J-lens adds little | jlens_vs_competitors.md |
| eval-aware replication (8B) | does paper's result carry over? | prereg E1a–E5 | breaks at eval vs real (0.50) | doesn't carry over | scripts/lens/evalaware_*.py, scripts/gen/evalaware_corpus.py; evalaware_replication.md |
| eval-aware models screen | which models voice "test"? | API, 10 models | none on blackmail; Qwen grader 20–66% | Claude-specific | evalaware_models_screen.md |
| steering | are J-lens directions causal? | prereg S1–S3, block 24 | J 47% vs logit 27%, probe 59% | causal, not best | scripts/lens/steer_answer.py; steering.md |
| point of no return | when does model commit? | prereg Q6, resampling | hints: before reasoning; 14B grader: mid | nulls partly task | scripts/explore/q6_chain.sh, run_forks; point_of_no_return.md |
| eval ablation 14B | removal test | prereg X1–X3 | talk 23%→0%, logit same, behaviour unchanged | Nanda's confound | evalaware_ablation_14b.md |
| C3 lead repeat | planted-claim lead | prereg R-C3, 389 runs | J 0.71, logit 0.60 | narrow positive | c3_lead_repeat.md |
| claim lead repeat | claim-follower lead | prereg R-CL | 0.54 fail | lead didn't replicate | claim_lead_repeat.md |
| 27B smoke | eval awareness in bigger model | Qwen3.6-27B, API + offload | prompts 0.64→0.55; before "testing" 0.49 | no | evalaware_big_smoke.md |
| literature: eval internals | are probes sound? | reading | key controls missing | — | docs/literature/evalaware_internals.md |
| literature: quirks | other belief-driven quirks | reading | Abdelnabi & Salem top | — | docs/literature/model_quirks.md |
| red team | claim strengths | reading | no lens claim solid | — | docs/overnight/redteam_claims.md |
| probe control (design) | do Nguyen/Hua probes read belief? | design only | 4 tests, ~4.7 GPU-h | pending user decisions | docs/overnight/design_probe_control.md |
| presentation | 10-min deck | Slides artifact | 14 + 4 slides | — | docs/presentation/slides.md |
| planned, not run | Hawthorne 14B; hack waterproofing (prereg W, 469543e); 27B full replication (prereg EA27, d05875e); claims firming (equivalence tests, item control, code review) | — | — | — | partial scripts: scripts/explore/hawthorne_*.py, scripts/gen/hawthorne_generate.py (uncommitted) |
