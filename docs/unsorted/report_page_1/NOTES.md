# Sources for the overnight report page (index.html)

Every number on the page comes from the files below. Paths are relative to `docs/overnight/`. Chart data was copied from the JSON files into the page (the `const D = {...}` object near the end of `index.html`), with values kept to 5 decimals; percentages are rounded the way the reports round them. Where a chart uses JSON and the report's markdown table differs by 0.01 (bootstrap reruns), the JSON value is shown.

## Masthead tiles

| Number | Source |
| --- | --- |
| −51 pp hint-wording effect; other factors < 7 pp | `reports/gap_factorial.md` step 2 effects; `data/gap_factorial.json` `factorial.effects.all.effects.*` |
| 67% vs 4% switches on known items | `reports/wave2_part2.md` §1 table; `data/wave2_part2.json` `wording.cells.think_qwen_hint_only_ours.switch`, `…think_qwen_ours_expert_ours.switch` |
| 2 of 1,231 thinking silent switches; 101 of 710 non-thinking | `reports/wave2_part2.md` §1 (silent switches) and §7; `log.md` 01:43, 02:36 |
| T2 J-lens AUROC 0.46, failed | `reports/lens_prereg.md` Results; `data/lens_prereg.json` `T2_preregistered_and_exploratory.pre_registered.jlens` |
| Re-elicitation 0.97 (instructed vs claim-persuaded) | `reports/lens_main.md` T7; `data/lens_main.json` `thinking.separation["kb_instr vs persuaded_claim"]["behaviour: re-elicited correct (1/0)"]` |

## Short version

Summaries of the three sections below; same sources. "M&B = Mirtaheri & Belkin" from `reports/wave2_part1.md` (E5-mb).

## Q1. Design

| Figure / number | Source |
| --- | --- |
| Pressure ladder: "Hint: The answer is X." 67%, M&B hint 60%, "Hint: An expert chose X." 13%, our pilot hint 4% | `data/wave2_part2.json` `wording.cells.*.switch` (`reports/wave2_part2.md` §1) |
| C3 planted start 30% [19, 42]; C2 5%, C1 4% | `data/wave2_part2.json` `e2_e4_e8.prefill.*.final_letter` (§2) |
| V2′ 24%, V2 19%, V0 2%, V1 0%, V4 2%, V5a 0%, V5b 3%, V3 99% correct | `data/wave2_part1.json` `grader.*.final_letter` / `final_correct` (`reports/wave2_part1.md` E3 table) |
| U2 12% [5, 20], U1 2% (X wrong) | `data/wave2_part2.json` `opinion_split.*.misaligned_follow` (§4) |
| P2 6%, P1 0% | `data/wave2_part2.json` `e2_e4_e8.pushback.*.final_letter` (§3) |
| Instructed I1 79%, I3 95% | `data/wave2_part1.json` `instructed.*.final_letter` |
| Grader variant matrix (information / incentive / illicit) | `reports/designs.md` §4 variants table |
| Grader details: open/rationalised counts, Jim Carrey item, logical_fallacies/31, meta flags 3–36 | `reports/wave2_part1.md` E3 table and finding 1; `data/wave2_part1.json` `grader.*.grader_outcome`, `meta_flag` |
| Moral small multiples: MoralChoice 84%, disputes 76%, scenarios 95% (M&B hint); U2 3/20/29%; V0 0%; null 0% | `data/wave2_part2.json` `moral_nudge_final_B` (§6) |
| Moral screen 39/40, 18/20, 14/20; 71 known items | `reports/wave2_part2.md` §6 screen table |
| Factual comparison bars: null 0% (wording items' unhinted P(B), 500 runs, 50 items), U2 12%, M&B 60%, V0 2% | `data/wave2_part2.json` `wording.unhinted_P_B`, `opinion_split.U2_sure.misaligned_follow`, `wording.cells.think_qwen_mb_expert_ours.switch`; `data/wave2_part1.json` `grader.V0_bug.final_letter` (same comparison row as §6) |
| Moral re-elicitation 51% / 23%, control 87%; silent 0 of 206 | `data/wave2_part2.json` `forks_reelicit_moral`; `reports/wave2_part2.md` §6 |
| Re-elicitation dumbbell (argmax correct, full history vs template) | `data/wave2_part1.json` `forks_reelicit_p1` (I1, I2, I3, V2, V2′); `data/wave2_part2.json` `forks_reelicit_p2` (wording_answer_is, wording_chose, C3, U2, P2; control 0.84) and `forks_reelicit_moral` (mb_expert) |
| Design flow, alternatives list (C3 30%; I3 95% name correct; I2 81% [71, 90]; P1/P3 0%, P2 6%; U1 2%, U2 12%; C1/C2 4–5%; E3 0–3%; not tried) | `reports/designs.md` §1–2, §5; `reports/wave2_part1.md`; `reports/wave2_part2.md` §2–4 |

## Q2. The gap

| Figure / number | Source |
| --- | --- |
| Main effects, all items: hint −51 [−56, −47], decoding −6, format −3, thinking +4; known_both hint −54 etc. | `data/gap_factorial.json` `factorial.effects.{all,known_both}.effects` |
| 16-cell strip (switch on known items, 43–73% vs 2–6%) | `data/gap_factorial.json` `factorial.cells.*.switch_known` (`reports/gap_factorial.md` step 2 table) |
| Backend check 58% [46, 70] vs 68% [57, 77]; 480/480 token-identical | `reports/gap_factorial.md` step 0 |
| Home cell 12% [8, 16] all, 4% [2, 7] known, pilot 1 in 450 | `reports/gap_factorial.md` bottom line |
| Extras 71% / 44% / 34% | `reports/gap_factorial.md` extras; `data/gap_factorial.json` `extras` |
| Interaction "thinking +8 [+4, +12] under M&B's hint" | `reports/gap_factorial.md` effects table |
| Wording ladder (10 arms) and paired contrasts (+47, +37, +19, +14, +9, +3, −7, 0, −7, +20, +3) | `data/wave2_part2.json` `wording.cells`, `wording.contrasts` (`reports/wave2_part2.md` §1) |
| Quotes from switches | `reports/wave2_part2.md` §1 reading 5 |
| Trajectories (P(B) by cut f, 11 cuts) | `data/wave2_part2.json` `forks_early_p2` keys `wording_answer_is|{hint,correct}`, `wording_chose|…`, `C3_think_prefill|…`, field `p_B`; `data/wave2_part1.json` `forks_early_mb.*.{hint,correct}.p_letter` (P(hinted letter)) |
| 0.82 at f = 0; letter mass 92–100% thinking, 15–35% M&B | `reports/wave2_part2.md` §5; `reports/wave2_part1.md` E5-mb caveat |
| 56% [50, 63] for M&B wording in our setting (Next, item 7) | `reports/gap_factorial.md` bottom line |

## Q3. Detection

| Figure / number | Source |
| --- | --- |
| J-lens definition, layers 14–32 | `CLAUDE.md` (lens); `reports/lens_pipeline.md` |
| Silent-switch meters: 2 / 1,231; 101 and 124 of 710 | `reports/wave2_part2.md` §1, §7; `data/wave2_part2.json` `nothink_scale` |
| Two thinking silent switches; +2.9 and +7.8 nats at L32 | `reports/wave2_part2.md` §1; `reports/lens_main.md` "Thinking-mode silent switches" |
| Margin chart (all groups, both lenses, layers 24/30/32, thinking and non-thinking) | `data/lens_main.json` `thinking.stats.<group>["corr/lp/pre_answer"]` (wrong-answer groups) and `["sal/lp/pre_answer"]` (salience groups); same for `non_thinking`; run and item counts from `*.groups.<group>.n_runs` and the stats `n_items`. Layer indices 5/8/9 of `layers` = 24/30/32. Matches `reports/lens_main.md` T1, T2, N1, N2 |
| |margin| ≤ 0.6 before the answer position; probe P(B) 0.97–1.0; CoT names correct in 98–100%, first 6–24% | `reports/lens_main.md` short answer and "Where the correct letter shows up" |
| AUROC panel "Instructed K→B vs claim-persuaded" | `data/lens_main.json` `thinking.separation["kb_instr vs persuaded_claim"]` (lens_main T7) |
| AUROC panel "Deferred vs persuaded" | `data/lens_main.json` `thinking.separation["deferred_claim vs persuaded_claim"]` |
| AUROC panel "Silent vs open" (0.65, 0.48, 0.45, 0.28, keyword 0.00 by construction) | `data/lens_main.json` `non_thinking.separation["silent_switch_all vs open_switch_all"]` (lens_main N6) |
| T2 panel (0.46, 0.43, L30–32 0.62, mid-CoT probe L20 0.70, CoT length 0.72, CoT names correct 0.48, template re-elicitation 0.82) | `data/lens_prereg.json` `T2_preregistered_and_exploratory.pre_registered`, `.exploratory.cot_baselines`, `.exploratory.mid_cot_probe_complied_vs_persuaded["L20 all 16 (mean)"]`, `exploratory_extra["T2 data, lens_main post-hoc L30-32 score …"]` |
| T1 panel (0.74, 0.59; 9 runs on 7 items; exploratory 0.63, 0.46) | `data/lens_prereg.json` `T1_preregistered`, `T1_exploratory_all_items_new_samples` |
| 115 of 124 silent switches on the excluded items | `reports/lens_prereg.md` T1 details |
| T1b definition and pass rule | `preregistration.md` T1b |
| T1b panel: J-lens S 0.542 [0.504, 0.577], logit 0.461 [0.424, 0.500]; 986 vs 7,268 runs, 333 vs 1,492 items, 1,744 known of 2,300 | `data/lens_prereg.json` `t1b.pre_registered`, `t1b.counts`; `reports/lens_prereg.md` T1b |
| T1b exploratory: length alone 0.54 [0.506, 0.572]; tertiles 0.53 / 0.51 / 0.57; L28 0.58, L32 0.50 | `data/lens_prereg.json` `t1b.exploratory.length_auroc_silent_vs_open`, `.within_length_tertile.T1–T3.jlens`, `.per_layer_auroc.jlens[7]`, `[9]` |
| T1b meter: 986 of 8,254 switches silent | `t1b.counts` |
| T1b reading ("almost correct" etc.) | `reports/lens_prereg.md` T1b Reading |
| "winner's curse" framing of 0.65 → 0.54 | orchestrator's instruction; the numbers are from `lens_main.md` N6 and `lens_prereg.md` T1b |

## Ledger, Next, Data, Housekeeping

| Item | Source |
| --- | --- |
| Ledger rows | the reports named in its Source column; counts: lens pipeline 1,500 + 84 + 120 runs (`lens_pipeline.md`), lens main 7,648 runs (`log.md` 02:45, `lens_main.md` data table), E5 962 + 629 traces, E7 1,420 + 1,976 forks (`log.md` 00:48, 02:14) |
| Next-step list | `reports/lens_prereg.md` "Next, if wanted"; `reports/wave2_part2.md` §8 (C3 readouts, patch); `reports/gap_factorial.md` "Not run"; `log.md` 21:48 (judge) |
| Data table: readout sets, run counts | `reports/lens_pipeline.md` (4 sets, 4.5 GB), `reports/lens_main.md` data table (14 GB thinking; set names and counts), `reports/lens_prereg.md` Files (w2_wording_lens 600 runs at 4.6 MB/run, w2_pushback_p2 200, nothink_scale copy), `reports/wave2_part2.md` §8 and §10 (octavia sets and generation paths) |
| "≈ 4.7 GB est." and "≈ 2.8 GB est." | computed for the page: 4,698 runs × 1.0 MB (non-thinking MB/run in `lens_pipeline.md`) and 600 × 4.6 MB (`lens_prereg.md`). Marked "est." on the page |
| Mayor lent for tonight only | `README.md` Machines |
| Mayor readouts ~20 GB to follow; w2 generation data 365 MB copied to octavia `data/from_mayor/explore/` at 03:05 | `log.md` 03:05 |
| Readouts being copied to octavia `data/from_mayor/` (w2_instructed, w2_grader, w2_wording_lens, w2_pushback_p2, gap_factorial_silent_think, analysis JSONs, explore/t1b runs); others regenerable in ≈ 2 GPU-h; moral readouts being produced on octavia, not analysed | orchestrator's status message at about 05:30 (not yet in `log.md`) |
| 18 min per 1,500 thinking runs | `reports/lens_pipeline.md` "Regenerating temp_check takes 18 min" |
| OpenRouter: 401 for the inherited key, user paused it, $0, 46/50, κ 0.89, four fixes pending | `log.md` 21:50, 21:48, 21:51; gold labels are Claude-agent labels, not human (`scripts/judge_eval/compare.py` heading) |
| known_items bug, 820 invalid runs, fix at 23:25, 200/200 check | `log.md` 23:16, 23:25; `wave2_patches.md` §0; `reports/wave2_part1.md` "Lost run" |
| Open `plan_run` patch | `wave2_patches.md` notes item 2; `reports/wave2_part2.md` §8 |
| Nothing committed | `README.md` rules; git status at the start of the session (all new code untracked) |

## Inconsistencies noticed in the sources

1. `log.md` 01:42 says the final wording-dissection rates "match the interim", but the interim numbers at 01:11 (70%, 50%, 36%) differ from the final ones in `wave2_part2.md` §1 (67%, 53%, 41%). The page uses the final report.
2. `wave2_patches.md` §0 still calls the `analyze_gap.py` patch "needed"; `log.md` 23:25 says it was applied and the pool regenerated.
3. C3 "recovered" is 134 runs in `wave2_part2.md` §2 and §8, but 61 in the §5 fork tables (apparently a matched comparison sample; the report doesn't say).
4. `wave2_part2.md` §5 says E7 used 963 traces; `log.md` 02:14 says 962 (for E5 and E7 together).
5. `log.md` 02:45 gives lens/probe AUROCs "0.35–0.62" for the headline contrast; `lens_main.md` gives 0.35–0.57 for K→B vs persuaded (0.62 is a contrast against honest errors).
6. A few AUROC intervals in `lens_main.md` T7 differ from `lens_main.json` by 0.01 (for example, probe pre-answer for K→B vs claim-persuaded: hi 0.52 in the markdown, 0.53 in the JSON).
7. `log.md` entries are not in time order (02:45 before 02:34; 03:10 before 02:58; 02:30 and 02:50 after 03:10).
8. E7 table in `wave2_part2.md` §5 reports a "B" share next to each "correct" share; the JSON has only `argmax_correct` and mean `p_B`, so the page plots only the correct share.

## T1b hook

T1b is now filled in. `index.html` has a hook near the top of its script: `const T1B = null;`. Replace `null` with an object and the T1b panel, the ledger row and the status chip switch from "pending" to the result, for example:

```js
const T1B = { kind: "fail", verdict: "Failed", summary: "J-lens S 0.xx [lo, hi] on N silent runs, M items",
  rows: [ { label: "J-lens S (the test)", r: 0.xx, lo: 0.xx, hi: 0.xx, cls: "ink" },
          { label: "Logit lens S", r: 0.xx, lo: 0.xx, hi: 0.xx, cls: "muted" } ] };
```

`kind` is one of `pass`, `fail`, `warn` (underpowered). Also update the T1b sentence in the Q3 lede, the short version and the masthead if the verdict changes the story.
