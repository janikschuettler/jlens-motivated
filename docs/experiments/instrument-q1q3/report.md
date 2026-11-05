# Instrument check: can any readout see the model's answer knowledge during its reasoning?

2026-11-03, arena9-octavia (L40S). Exploratory. Derived numbers: [../data/instrument_q1q3.json](data/instrument_q1q3.json) (curves as means; full CIs per layer × position in octavia `data/readouts/instrument_q1q3.json`). Figure: [../../findings/figs/fig10_instrument.svg](fig10_instrument.svg).

**Result: no readout of the reasoning itself passes the check.** Mid-CoT, the J-lens and logit lens (letters or content words) and two trained probes do not point to the correct option any better than the CoT text does, even in unhinted runs that go on to answer correctly. Rates are 0.24–0.40, against chance 0.25 and 0.38 for a text baseline. The one readout that works is a probe on the prompt's *option text*, before any reasoning. It points to the correct option on 72% [65, 78] of unhinted runs, 62% [47, 77] of instructed "knows A, argues B" runs and 26% [7, 48] of honest errors. The J-lens adds nothing over the logit lens anywhere.

*Interpretation:* the model's knowledge of the answer is linearly readable at the option tokens. During the reasoning, none of our readouts recovers it beyond what the text already says. So the lens nulls during reasoning (lens_main T4/T5) say nothing about whether the knowledge is still there. The option-text probe measures the item, not the run (see caveats), so it can't tell a run that knows from a run that doesn't on the same item. Moderate confidence. Small honest-error group (14 items, 8 with content tokens), and the option-text probe was added after the first results.

## Question and design

Last night's lenses showed the answer letter only at the final answer position, even in unhinted correct runs (lens_main T4/T5). So a lens null in followed runs may reflect the instrument, not the model. An instrument for "does the model still know A while it argues B" must pass two checks:
- **sensitivity:** it points to the correct option where we know the model knows it, including runs where it argues for B;
- **specificity:** it does not point to the correct option where the model doesn't know it.

"Points to X" means that X has the highest of the four option scores. Ties share the point. Chance is 0.25.

**Groups** (thinking mode, home prompt format "ours", T 0.6; known = ≥ 9/10 correct in the gap screen's setting A):

| Group | Source | Runs / items | Role |
| --- | --- | ---: | --- |
| unhinted | screen A known items, final correct, 3 per item (seed 0), + w2 unhinted | 572 / 164 | training data and reference (probes: held-out folds) |
| kb_instr | w2 instructed I1 (argue) + I3 (argue, conceal), final = the instructed B | 348 / 39 | sensitivity |
| i2_knows | I2 (debate + report), argues B, final "Answer:" = correct | 162 / 38 | sensitivity |
| i2_says_b | I2, final B | 32 / 13 | sensitivity (small) |
| honest_error | screen A items with 2–8/10 correct, final wrong | 62 / 14 | specificity |
| unstable_correct | same items, final correct (≤ 3 per item) | 50 / 18 | same-item control |
| wording_switch / resisted | w2 wording arms, known items, final B / correct | 193 / 23, 125 / 21 | for later use |
| c3_kept_b / recovered | w2 C3 planted claim, final B / correct | 35 / 12, 60 / 15 | for later use |

Probe training also used 1,140 more unhinted correct screen runs on the same known items (block outputs only). Median CoT length differs a lot by group: unhinted 749 tokens, kb_instr 1,565, i2_knows 1,470, honest errors 2,542, unstable_correct 3,024.

**Positions.** These are the 16 evenly spaced CoT anchors (relative position k/15 of the span inside `<think>…</think>`), `</think>` and `pre_answer`. **Mid-CoT** = anchors 4–11 (relative 0.27–0.73) and **late** = anchors 12–15. These windows were fixed before looking. Comparing at matched *relative* positions is the length control. I also split by CoT-length tertile (below).

**Readouts** (layers 14–32 for the lenses; per readout, the headline layer is the one with the best held-out unhinted rate mid-CoT, chosen on the reference group only):

1. **Letter tokens** (existing): J-lens / logit lens / the model's own output, log-sum-exp over " X", "X", "(X". "_bin" = mean over the positions of CoT bin k of 16 instead of the single anchor.
2. **Content-word tokens** (new, [configs/content_tokens.json](../../../configs/content_tokens.json), fixed before the readout). For each option, I took the first content word that no other option of the item contains, and the first token of " word", "word" and " Word". Each ID was checked as a single-token round trip. A "word" or " Word" token under 2 characters was left out of the option's set.
   - 96 of 200 items qualify. 104 were excluded: 78 have an option without a distinctive content word (mostly numeric options), 23 have a " word" token under 2 characters (" U" for "Ursa"), and 3 share a first token between options.
   - Only 302 of 384 " word" forms are whole words; the rest are fragments such as " hyp" or " Dynam".
   - **Calibrated ("bincal", post hoc):** each option's score minus its mean over the 64 stored prompt positions before the reasoning. This removes token-frequency offsets: uncalibrated, a frequent word such as " much" wins the argmax everywhere.
3. **Letter probe** (as asked): 4-class logistic regression (standardised, C = 0.05) on the block output at one layer × position, label = correct letter. It was trained on unhinted correct runs only. I used 5-fold item-level splits; every run is scored by the model that held out its item. "Pooled" = one probe per layer trained on all 16 CoT anchors.
4. **Post hoc, added after the first results:**
   - **Option-matching probe:** score_i = w·(z(h_t) ⊙ z̃(o_i)), where o_i is the mean block output over option i's text in the prompt and z̃ is centred over the item's options. It is letter-agnostic and uses the same training data and splits. Layers 16, 20, 24 and 28.
   - **Option-text probe:** the same model with z̃(o_i) alone, i.e. the prompt's representation of each option, with no CoT state.
5. **CoT text baseline:** the option named last ("text_last") or most often ("text_count") in the CoT up to the position. Names are letter patterns as in lens_main, the option's text, or its content word.

Rates are item-level means (each item counts once). 95% CIs come from 2,000 item resamples.

## Q1: sensitivity and specificity along the CoT

Share pointing to the correct option, mid-CoT (headline layer). Content-word rows use only items that have content tokens (80 / 19 / 18 / 8 / 8 items in the five columns).

| Readout | unhinted (ref.) | kb_instr (sens.) | i2_knows (sens.) | honest errors (spec.) | unstable_correct (same items) |
| --- | --- | --- | --- | --- | --- |
| letter probe, L18 | 0.30 [0.27, 0.33] | 0.26 [0.22, 0.30] | 0.26 [0.21, 0.30] | 0.22 [0.14, 0.32] | 0.26 [0.16, 0.37] |
| option-matching probe, L24 | 0.30 [0.28, 0.32] | 0.27 [0.24, 0.29] | 0.28 [0.25, 0.32] | 0.28 [0.22, 0.36] | 0.24 [0.17, 0.30] |
| J-lens letters (bin), L32 | 0.28 [0.24, 0.32] | 0.24 [0.18, 0.31] | 0.27 [0.22, 0.33] | 0.34 [0.19, 0.51] | 0.33 [0.20, 0.49] |
| J-lens content words (cal.), L26 | 0.40 [0.34, 0.46] | 0.40 [0.28, 0.53] | 0.35 [0.20, 0.51] | 0.19 [0.02, 0.41] | 0.18 [0.00, 0.44] |
| logit lens content words (cal.), L26 | 0.39 [0.33, 0.46] | 0.40 [0.27, 0.54] | 0.39 [0.23, 0.57] | 0.06 [0.01, 0.15] | 0.05 [0.00, 0.15] |
| CoT text, last named | 0.38 [0.35, 0.41] | 0.30 [0.26, 0.35] | 0.29 [0.23, 0.34] | 0.44 [0.32, 0.56] | 0.49 [0.38, 0.60] |
| option-text probe (prompt), L24 | 0.72 [0.65, 0.78] | 0.62 [0.47, 0.77] | 0.68 [0.53, 0.82] | 0.26 [0.07, 0.48] | 0.30 [0.11, 0.50] |

- **The trained CoT probes fail on their own reference group.** On held-out unhinted runs they are at 0.28–0.31 at every layer mid-CoT. They rise only where the CoT has concluded: at the last CoT token the letter probe gives 0.85 [0.81, 0.89], at `</think>` 0.48, and at `pre_answer` 1.00 at L24–32. In instructed runs it reads B there (P(points to B) 0.99), as lens_main found. *Interpretation:* these probes read the conclusion the text has reached, not knowledge held in reserve. High confidence for these probe types; a nonlinear or differently trained probe might do better.
- **The letter lenses are at chance mid-CoT** in every group (0.22–0.34, all CIs cover 0.25). *Interpretation:* as in lens_main, the letter readout asks "is a letter the next token?", which mid-CoT it isn't.
- **Calibrated content words point to the correct option a bit above chance** in unhinted (0.40) and instructed runs (0.40, while the model argues B). Honest errors score low (0.19 J-lens, 0.06 logit). The CoT text gives 0.38 / 0.30 / 0.44 in the same groups.
  - This looks like sensitivity and specificity, but the same-item control removes it. On the honest-error items, the runs that answer *correctly* score just as low (0.18 J-lens, 0.05 logit).
  - So the low honest-error score belongs to those 8 items' tokens, not to the runs' knowledge state.
  - In instructed runs the content readout (0.40) is a little above the text (0.30), but the CIs overlap.
  - *Interpretation:* no evidence that the content readout tracks knowledge rather than which option the text is about to discuss. Low confidence either way (19 and 8 items).
- **The option-text probe separates known items from unstable ones.** AUROC of its correct-option score: kb_instr vs honest errors 0.73 [0.55, 0.87], i2_knows vs honest errors 0.78 [0.64, 0.90], unhinted vs honest errors 0.80 [0.70, 0.89]. On content items: 0.89 [0.73, 0.99]. It is at 0.74–0.83 in the wording and C3 groups too, whatever the run did.
  - The CoT probes give 0.43–0.56 on the same contrasts. The calibrated content lenses give 0.60–0.67, with CIs from about 0.38 to 0.84.
  - *Interpretation:* the model's representation of the options already marks the correct one on items it knows, and doesn't on items it gets wrong half the time. It is an item-level knowledge readout. Moderate confidence (14 honest-error items).

**At `</think>`** nothing points to the correct option in instructed runs: the probes give 0.20–0.23, the content lenses 0.34–0.42 (wide CIs) and the text 0.19. In unhinted runs, the text (0.69) beats every internal readout there (probes 0.45–0.48, lenses ≤ 0.26).

**Length.** Within each CoT-length tertile (cuts at 945 and 1,786 tokens), the CoT letter probe stays at 0.16–0.32 mid-CoT for unhinted, kb_instr and i2_knows (none above 0.32). Length doesn't hide a signal.

## Q3: where the probe sees the answer and the J-lens doesn't

- **Before the reasoning, at the option tokens:** a trained probe reads the correct option on known items, including the items of instructed runs (62–79%). That knowledge is represented. We didn't read the J-lens at the option positions, so I can't say whether the J-lens would verbalise it there. Only the last 64 prompt positions are stored, and these rarely cover the options.
- **During the reasoning:** neither the probes nor the J-lens see it. Mid-CoT, the probes do no better than the J-lens content readout (0.26–0.30 against 0.40 in instructed runs). There is no case where a mid-CoT probe sees the correct answer and the J-lens doesn't.
- **J-lens vs logit lens:** the same within CI everywhere. Content words mid-CoT: 0.40 vs 0.39 (unhinted) and 0.40 vs 0.40 (kb_instr).
- *Interpretation:* the knowledge is represented, at the options, in a form a linear probe reads. Whether it stays represented during the reasoning, our instruments can't tell. A probe that reads it at the options doesn't transfer to CoT positions, and the lenses only show what the text is about to say.

## Caveats

- **Item, not run.** The prompt's option tokens come before the instruction or hint, so the option-text probe has the same value for every run on an item. Its "specificity" means "items the model knows less well". It does not mean "runs where the model doesn't know". The same holds for any group difference on different items: kb_instr (39 items) and honest errors (14) are different items.
- **Post hoc.** The content-word calibration, the option-matching probe and the option-text probe were added after the first results (uncalibrated lenses and letter probe at chance). The mid-CoT window and the layer rule were fixed before.
- **Calibrated letters are confounded** in instructed runs: the prompt tail contains "argue that the answer is B", so subtracting it penalises B. Calibrated letter readouts are in the JSON (e.g. the model's own letters 0.43 in kb_instr), but I don't use them.
- **Small groups:** 14 honest-error items (8 with content tokens), and 4 i2_says_b content items.
- **Content tokens cover 96 of 200 items.** Many are word fragments. Numeric options are excluded.
- The probes are linear, use one position at a time, and were trained on 1,712 runs from 164 items. RFM-style probes (M&B) were not tried.
- The text baseline is a regex heuristic.

## Readouts, checks and files

- **Readout extension.** `readout_runs.py --content-tokens configs/content_tokens.json` adds per-run arrays `x_ids`, `x_opt`, `x_lp`, `x_rank`, `x_model_lp` and `x_model_rank`; the default arrays and format are unchanged. Token IDs are re-checked against the tokenizer, and a run on an item without tokens stops the readout. `--validate` and the rank-vs-top-25 check cover the new arrays. Code: `src/jmr/lens/batch.py`, `scripts/lens/readout_runs.py`, `scripts/lens/content_tokens.py`.
- **Readouts (octavia `data/readouts/`):**
  - `instr_ct_core` (607 runs, with content tokens), `instr_nc_core` (619) and `instr_ct_later` (413, wording + C3), all `--cot-resid`;
  - M0.4: median |Δ| 0.0003–0.0011, p99 0.133–0.135 nats, all pass;
  - every rank / top-25 overlap is exact: letters 113,045, content tokens 21,981;
  - the 4 validated runs are bit-identical to the single-run readout, content tokens included;
  - 4.3–5.8 MB per run, 36 / 31 / 40 runs per minute with two processes;
  - `instr_nc_later` (382 wording/C3 runs without content tokens) was not read out, to keep ≥ 10 GB free.
- **Block outputs only** (`scripts/lens/cot_resid_runs.py`, same forward and hooks):
  - `instr_train_extra`: 1,143 runs; 3 check runs bit-identical to `instr_ct_core`, 12/12 arrays;
  - `instr_options` (`--options-only`): mean block output over each option's text at layers 16, 20, 24 and 28, 2,779 runs. Each span is checked to decode to the item's option text.
- **Run selection:** `scripts/lens/instrument_select.py` → octavia `data/instrument/`.
- **Analysis:** `scripts/lens/analyze_instrument.py` (CPU, ~7 min with 64 processes): `uv run --project envs/lens python scripts/lens/analyze_instrument.py --out data/readouts/instrument_q1q3.json --docs-out docs/experiments/instrument-q1q3/data/instrument_q1q3.json`.
- **Disk:** about 10 GB in `data/readouts/instr_*`; octavia had 11 GB free at the end.

## Next steps

- **A run-level knowledge readout needs runs that differ within an item.** Options:
  - read the option-text probe's direction at CoT positions, i.e. apply the option-position probe to h_t;
  - train on resisted vs followed runs of the same item (M&B style; confounded with the outcome);
  - track how strongly the model attends to or re-represents the correct option's tokens.
- **Read the J-lens at the option positions.** This needs a readout with a longer prompt tail (rule 3: re-run). It would test Q3 directly: does the J-lens verbalise, at the options, what the option-text probe reads?
- **More honest-error items** (2–8/10) for specificity: the screen has only 18.
