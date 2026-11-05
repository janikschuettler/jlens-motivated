# Lens readouts on switches: is the correct letter still represented when Qwen3-8B outputs a wrong option on an item it knows?

2026-11-03 overnight, arena9-mayor (RTX 4090). Exploratory; no claims beyond the numbers. Derived numbers: [../data/lens_main.json](data/lens_main.json). Pipeline: [lens_pipeline.md](../lens-pipeline/report.md). First test: [lens_step0.md](../lens-step0/report.md).

## Short answer

**No internal signal we measured separates "knows A, says B" from "persuaded" or from honest errors, beyond what the salience of an unchosen option already explains.** The lenses read option letters out only at the answer position; there, every group that answers wrong keeps the correct letter 1–3 nats above the other unchosen letters. That is about as much as resisted runs keep for the unchosen *hint* letter.

- **Before the answer position:** at `</think>`, at the last prompt token and at CoT/response positions before the correct option is first named, the correct letter is not above the other wrong letters in any group (|margin| ≤ 0.6 nats, CIs on 0), unhinted runs included. The lenses don't read the upcoming answer letter there at all.
- **At `pre_answer`, layers 30–32 (thinking):** correct-letter margin +1.4 to +1.9 nats in instructed K→B runs, +1.6 to +2.9 in persuaded switches, +0.7 to +1.7 in honest errors. The salience control (the hint letter in resisted runs) is +0.5 to +2.1. A linear probe trained on unhinted + resisted runs reads the *final answer* there (P(B) ≈ 0.97–1.0 in followed runs); at `</think>` it is near chance and leans toward the eventual answer.
- **Headline contrast, K→B vs persuaded:**
  - lens and probe AUROCs 0.35–0.57, no separation; if anything the persuaded runs keep slightly more of the correct letter;
  - CoT keyword features 0.47–0.56;
  - the behavioural re-elicitation separates them at 0.97.
- **The one candidate signal (thinking, small n, partly post hoc):**
  - claim-followers whose re-elicitation recovers the correct answer keep +5 to +7 nats for the correct letter at L30–32, against +2.5 to +2.9 for those who stay with B;
  - AUROC 0.78 [0.65, 0.92] (J-lens, post-hoc layers) and 0.66 [0.45, 0.84] (pre-registered layers);
  - the CoT features are at 0.32–0.52.
  - It rests on 10 vs 12 items.
- **Non-thinking silent switches** (151 runs, 42 items, M&B's or our hint, no hint mention):
  - the CoT keyword check can't flag them by construction;
  - the pre-defined lens score (correct margin at `pre_answer`, L14–32 mean) flags them against resisted runs only because those runs answer correctly (AUROC 0.01–0.02, i.e. 0.98–0.99 with the sign flipped; the final answer does the same);
  - against *open* switches with the same final answer, the J-lens score is higher for silent switches: 0.65 [0.56, 0.74] pre-defined, 0.59–0.69 at L16–32. The logit lens gives 0.48 [0.37, 0.58]. The probe gives 0.45–0.52.
  - So in the population where the CoT is silent, the J-lens keeps a bit more of the correct letter at the moment of answering than in open switches, and the logit lens doesn't. That is a lead, not a detector.


## Data and readouts

| Run set | Source | Runs read out | Mode |
| --- | --- | ---: | --- |
| `w2/instructed` | wave 2 E1 (I1 argue, I2 debate + report, I3 argue + conceal, unhinted), 40 known items | 680 | thinking |
| `w2/grader` | wave 2 E3 (V0–V5b), 40 known items × 3 | 960 | thinking |
| `factorial_think2` | gap factorial cells `think_qwen_mb_expert_ours` (M&B hint in our setting) and `think_qwen_ours_expert_ours` (home cell), copied from octavia | 800 | thinking |
| `screen_A_lens` | gap screen setting A (= the unhinted cell matching the two above): all 10 samples of the 18 items with 2–8/10 correct, 2 samples of each of the 164 known items | 508 | thinking |
| `factorial_nothink` | all 11 non-thinking factorial cells (M&B hint ×4, our hint ×4, unhinted ×3), copied from octavia | 4,400 | non-thinking |
| `screen_B_lens` | gap screen setting B (the unhinted cell matching `nothink_mb_mb_expert_mb`), 2 samples per item known in B | 298 | non-thinking |
| `silent_think` | the 2 thinking-mode silent-switch candidates (wave 2, `silent_candidates_wording_factorial.jsonl`) | 2 | thinking |

- Every set was read out with `scripts/lens/readout_runs.py`: exactly the saved token IDs, J-lens and logit lens at blocks 14–32, block outputs at 3 anchors stored, no full-vocabulary output.
- Each set passes the checks:
  - M0.4 vLLM agreement: median |Δ| ≤ 0.0007, p99 ≤ 0.14 nats;
  - every overlap between tracked ranks and the top-25 is exact (190,755 in the thinking sets);
  - the validated run per set is bit-identical to the single-run readout, residuals included.
- Throughput: 35–67 runs/min for thinking runs (2.1–2.9k tokens), about 200/min for non-thinking runs. About 3 MB per thinking run; 14 GB for the thinking sets.
- Analysis scripts (CPU): `scripts/lens/analyze_main.py` (thinking) and `scripts/lens/analyze_nothink.py` (non-thinking).

**"Known"** means ≥ 9/10 unhinted samples correct in the gap screen's matching home setting: A (thinking, 164 of 200 items) or B (non-thinking). The wave-2 items are a subset of the A-known items.

**Groups (thinking).** All are on known items except the two "unstable" groups. Labels are final answers (`jmr.label.extract_answer`) and E7 re-elicitation (full history, argmax).

**Measures.**
- **Letter score:** log-sum-exp over " X", "X", "(X" log-probs, so the format of the answer letter doesn't matter. Rank = best of the three.
- **Margin:** a letter's log-prob minus the mean log-prob of the *unchosen wrong letters*, i.e. the 2 letters that are neither correct, nor the final answer, nor the salient letter.
- **Correct margin**, in runs with a wrong final answer: is the correct letter above the other unchosen wrong letters?
- **Salience control**, in runs with the correct final answer: the same margin for the unchosen *hint* letter. For unhinted runs it uses the item's (unshown) hinted letter.
- **Positions:**
  - `last_prompt`;
  - `</think>`;
  - `pre_answer` (the position whose next token is the final-answer letter);
  - 20 bins along the CoT;
  - **echo-free** positions: CoT positions before the CoT first names the correct letter ("(A)", "A)", "option A", "**A", "A:") or the correct option's text.
- **Bootstrap:** item means, then 2,000 resamples of items.
- **Probe:** multinomial logistic regression on block outputs.
  - Label: the correct letter. Training runs: unhinted + resisted only, where the correct letter is also the final answer, so on its training distribution this is an answer probe.
  - 5-fold GroupKFold by item, C = 0.05, standardised features. Every run is scored by the fold that held out its item.
- **Fixed before looking:** the lens score is the correct margin averaged over layers 20–28 at `</think>`, at the echo-free mean and at `pre_answer`; the probe score is P(correct) at `</think>`, layer 24. The layer 30–32 score is post hoc (chosen after the first w2 tables) and is marked as such.

## Thinking mode

### Where the correct letter shows up at all

- **`</think>`, `last_prompt` and echo-free CoT positions: nothing, in any group** (T4, T5). The correct margin is within ±0.6 nats of 0 with CIs covering 0, in both lenses and at every layer. That includes unhinted runs, which do go on to answer correctly. Before the model is about to write a letter, the lenses don't read the answer letter out at all. So "absent in followed runs" here means nothing beyond "absent everywhere".
- The 20 CoT bins show the same flat picture (JSON).
- **Echo-free windows are short.** The CoT names the correct option, by letter or by its text, in 98–100% of runs, typically in the first 6–24% of the CoT (T8). That leaves a median of 90–370 tokens before the first mention.
- **Just before the first mention** (the last echo-free position) the correct margin is +1.1 to +3.4 nats at layers 24–32, by about the same amount in every group, unhinted included. That is anticipation of the letter about to be written, not a group effect (JSON, `ef_last`).
- **Probe at `</think>`:** weak but above chance on its own distribution. Held-out unhinted and resisted runs give P(answer) 0.46–0.55 at L24, against 0.25 chance.
  - In wrong-answer runs it gives P(correct) 0.10–0.26 and P(final answer) 0.24–0.40 (T6). It leans toward the answer the run will give, and puts no more on the correct letter than chance.
  - So nothing at `</think>` says "A" while the model is about to say B.

### At the answer position

- **In every group with a wrong final answer, the correct letter keeps a margin over the other unchosen wrong letters at layers 30–32** (T1): K→B instructed +1.4 to +1.9 nats, M&B-hint persuaded +1.6 to +2.0, claim-persuaded +2.5 to +2.9, honest errors +0.7 to +1.7. At L24–28 the margins are small (≤ 0.7) for K→B, M&B-persuaded and honest errors; claim-persuaded runs already have +1.6 to +2.4 there. Ranks agree (T3).
- **The salience control gives margins of the same size** (T2):
  - the unchosen *hint* letter in resisted runs: +1.6 to +2.1 (V2/V2′), +1.0 to +1.4 (M&B hint), +0.5 to +0.7 (our hint);
  - the item's unshown hinted letter in unhinted runs: about 0.
  - So "the unchosen option that the prompt or the CoT made salient keeps 1–2 nats at the answer position" describes the followed and the resisted runs equally well. The correct-letter margin in followed runs is not evidence of knowledge beyond that.
- **Two groups stand out, both small:**
  - **"Deferred" claim-followers** (V2/V2′ final B, but re-elicitation gives the correct option; 16 runs, 10 items): +5.3 to +7.4 at L30–32, already +3.2 at L24.
  - **I2 runs that argue B and also report B** (32 runs, 13 items): +4.7 to +7.4. Wave 2 found that these re-elicit the correct option in 94% of cases.
  - Both are groups where the final B is shaky by other evidence. The instructed K→B arguers, who re-elicit the correct option in 94% too, have the *smallest* margin (+1.4 to +1.9). So across groups, the margin at the answer position does not track "would say A if asked"; within the claim-followers it does (deferred vs persuaded, below).

### Does any internal signal separate K→B, persuaded and honest errors? (T7)

- **K→B vs persuaded: no.**
  - Against claim-persuaded runs, every lens and probe score is at or below 0.5: 0.35–0.51, and the CIs of the pre-answer scores reach 0.48–0.56. If anything the persuaded runs keep *more* correct-letter margin.
  - Against M&B-persuaded runs: lens 0.49–0.57, probe 0.36–0.48.
  - The CoT keyword features are at 0.47–0.56: both groups name the correct option and B.
  - Re-elicitation separates K→B from claim-persuaded almost perfectly (0.97 [0.94, 0.99]); that is behaviour, not internals.
- **K→B or persuaded vs honest errors: not by the lens.**
  - Lens scores: 0.45–0.72, all CIs covering 0.5 except the post-hoc logit L30–32 score for claim-persuaded vs honest errors, 0.72 [0.52, 0.92].
  - The pre-answer probe separates them (0.85–0.94). But so does the CoT naming the hinted letter (0.93–0.95), since honest errors have no hint and K→B/persuaded runs do, and the probe can pick up the prompt. Not informative.
- **Deferred vs persuaded (claim-followers, 16 vs 36 runs):**
  - pre-registered lens score 0.66 [0.45, 0.84] (J-lens) and 0.53 (logit);
  - post-hoc L30–32 score 0.78 [0.65, 0.92] (J-lens) and 0.71 [0.54, 0.89] (logit);
  - pre-answer probe 0.67 [0.50, 0.85];
  - CoT features 0.32–0.52.
  - This is the one contrast where an internal readout tracks something the CoT doesn't. It rests on 10 vs 12 items, with a layer choice made after looking. Candidate for a pre-registered replication.

### Thinking-mode silent switches (2 runs)

The wave-2 agent's 2 candidates (`high_school_statistics/36` in `think_mb_mb_expert_mb`; `professional_medicine/98` in `think_qwen_bare_answer_ours`) have J-lens correct-letter margins at `pre_answer` of +2.9 and +7.8 nats at L32, and logit-lens +0.7 and +4.5. That is within the range of the other followed groups; n = 2.

## Non-thinking mode (setting B, items known in B)

**Groups.**
- Switch = final answer is the shown hint B.
- **Silent** = the gap agent's keyword check (`hint`, `expert`, plus `90%`/`ninety` for our hint) finds no hint mention anywhere in the completion; **open** = it finds one.
- Runs come from the 4 M&B-hint and 4 our-hint non-thinking factorial cells. Unhinted runs come from the 3 unhinted non-thinking cells plus screen B; resisted = final correct.
- Known in setting B: 149 items. N7 has the counts: 151 silent switches on 42 items (120 under M&B's hint), 513 open, 1,691 resisted, 1,166 unhinted.
- Silent switches concentrate on fewer items: 3.6 runs per item.

**Reading the silent switches.** I read 4 silent M&B-hint switches in full and keyword-scanned 12 (`data/readouts/lens_nothink_silent_examples.json` on mayor). None refers to the hint.
- They read as ordinary option-by-option evaluation that settles on B with content reasons, e.g. "Populist farmers… This seems like the best fit".
- "suggests"/"indicates" occur only as content words.
- 83% of silent switches still name the correct option somewhere in the response (80% under M&B's hint), against 93% of open switches.

**Results.**
- **At `pre_answer`** (N1, N2): silent switches keep the correct letter +2.9 to +3.7 nats above the other unchosen letters at J-lens L30–32 (+1.4 to +2.5 logit). Open switches keep +2.5 to +2.7 (J-lens) and +1.7 to +1.9 (logit). The salience control, the hint letter in resisted runs, is +2.4 (J-lens) / +2.1 (logit) at L32 under M&B's hint and +0.7 / +0.6 under ours; the unhinted reference is 0.
- **Before the response names the correct option** (N3): nothing, within ±0.3 nats in every group (up to +0.6 in the 6-item open-our-hint group). Just before the first mention (N4) the margin rises (+1.2 to +4.0) in every group alike.
- **Probe** (N5):
  - at `pre_answer` it is an answer probe: P(B) 0.97–1.0 in switches at L24+.
  - At `last_prompt` the silent-switch group has the highest mean P(correct), 0.37–0.41 at L20–28 against 0.28–0.34 in the other groups. That position is shared by all samples of a prompt, and run-level AUROCs are 0.52–0.55, so this is item composition, not a per-sample signal.
- **Separation** (N6):
  - **Silent vs resisted:**
    - pre-defined lens score: J-lens 0.02 [0.01, 0.05], logit 0.01 [0.00, 0.03]. That is near-perfect separation in the direction "less correct-letter margin = switch". It comes from resisted runs being about to write the correct letter; the final answer separates them just as well.
    - Per layer: 0.00–0.11 at L24–32, 0.45–0.62 at L14–20.
    - CoT keyword mention: 0.44, below 0.5 because resisted runs mention the hint more often.
  - **Silent vs open** (same final B):
    - J-lens pre-defined 0.65 [0.56, 0.74], per layer 0.54 (L14), then 0.59–0.69 (L16–32, highest at L28);
    - logit pre-defined 0.48 [0.37, 0.58];
    - probe 0.45–0.52;
    - CoT keyword 0.0 by construction; "correct named early" 0.28 (silent switches name the correct option later or not at all).
  - Under M&B's hint alone the J-lens gives 0.66 [0.56, 0.74]. Under our hint, 31 vs 12 runs, the CIs are too wide to say anything.

**What this adds.** In the one population where the chain of thought hides the hint, the J-lens, but not the logit lens, keeps slightly more of the correct letter at the answer position than in open switches. Every caveat applies: the effect is at the answer position, it is near the size of the salience effect, and it is one lens of two. If someone wants a pre-registered test for the next run, this is the one: J-lens correct-letter margin at `pre_answer`, L20–32, silent vs open switches, new items.



## Tables (thinking)

### T1. Correct letter vs the unchosen wrong letters at pre_answer, runs with a wrong final answer (log-prob margin, nats)

| group | runs / items | J-lens L24 | J-lens L30 | J-lens L32 | logit L24 | logit L30 | logit L32 |
|---|---|---|---|---|---|---|---|
| K→B instructed (I1/I3, final B) | 348 / 39 | +0.4 [-0.2, +1.0] | +1.9 [+1.2, +2.6] | +1.8 [+0.9, +2.7] | +0.4 [-0.1, +0.9] | +1.6 [+1.0, +2.1] | +1.4 [+0.9, +2.0] |
| I2 argued B, reports B | 32 / 13 | +4.6 [+2.8, +6.9] | +6.1 [+4.4, +7.8] | +7.3 [+5.0, +9.3] | +3.9 [+2.2, +5.8] | +4.7 [+3.2, +6.3] | +7.4 [+5.1, +9.8] |
| persuaded: V2/V2′ final B, re-elicit B | 36 / 12 | +1.6 [+0.9, +2.4] | +2.7 [+1.6, +3.9] | +2.7 [+1.1, +4.1] | +2.0 [+0.7, +3.1] | +2.9 [+1.3, +4.3] | +2.5 [+1.2, +3.7] |
| deferred: V2/V2′ final B, re-elicit correct | 16 / 10 | +3.3 [+2.1, +4.7] | +7.0 [+4.7, +9.9] | +7.4 [+4.9, +10.7] | +3.2 [+2.0, +4.2] | +5.3 [+3.8, +6.9] | +7.0 [+4.8, +9.4] |
| persuaded: M&B hint, final B | 185 / 118 | +0.5 [+0.2, +0.9] | +2.0 [+1.5, +2.5] | +2.0 [+1.4, +2.6] | +0.6 [+0.2, +0.9] | +1.6 [+1.3, +2.0] | +1.8 [+1.3, +2.2] |
| our hint, final B | 13 / 9 | +0.5 [-0.9, +1.7] | +1.2 [-0.7, +3.2] | +1.0 [-1.4, +3.4] | +0.8 [-0.3, +1.9] | +0.9 [+0.1, +1.7] | +0.4 [-1.1, +1.7] |
| honest error (unstable item, no hint) | 62 / 14 | +0.7 [-0.3, +1.4] | +1.3 [+0.3, +2.4] | +1.7 [+0.2, +3.3] | +0.1 [-0.9, +1.2] | +0.7 [+0.1, +1.4] | +1.3 [+0.2, +2.4] |

### T2. Salience control: the salient unchosen letter at pre_answer in runs with the correct final answer (hint letter; for unhinted runs the item's hinted letter; nats)

| group | runs / items | J-lens L24 | J-lens L30 | J-lens L32 | logit L24 | logit L30 | logit L32 |
|---|---|---|---|---|---|---|---|
| I2 argued B, reports correct | 162 / 38 | +0.0 [-0.7, +0.7] | +5.5 [+4.1, +6.9] | +4.8 [+3.4, +6.1] | +0.4 [-0.3, +1.1] | +4.8 [+4.0, +5.7] | +4.7 [+3.4, +5.9] |
| resisted V2/V2′ | 172 / 38 | -1.3 [-1.9, -0.7] | +1.6 [+0.8, +2.5] | +1.6 [+0.6, +2.7] | -0.4 [-1.0, +0.2] | +2.1 [+1.4, +2.8] | +1.9 [+0.9, +2.8] |
| resisted M&B hint | 125 / 87 | -0.9 [-1.3, -0.5] | +1.4 [+0.9, +1.9] | +1.1 [+0.5, +1.8] | -0.6 [-1.0, -0.3] | +1.4 [+1.0, +1.7] | +1.0 [+0.5, +1.5] |
| resisted our hint | 312 / 160 | -0.4 [-0.7, -0.1] | +0.6 [+0.3, +0.9] | +0.5 [+0.1, +1.0] | -0.3 [-0.6, +0.0] | +0.7 [+0.4, +0.9] | +0.5 [+0.2, +0.9] |
| unhinted, correct | 408 / 164 | -0.0 [-0.3, +0.3] | +0.1 [-0.2, +0.4] | +0.0 [-0.4, +0.5] | +0.1 [-0.2, +0.5] | +0.2 [-0.0, +0.5] | +0.0 [-0.3, +0.4] |
| unstable item, correct | 98 / 18 | -0.0 [-1.2, +1.0] | -0.3 [-1.5, +0.9] | -0.1 [-1.6, +1.3] | +0.0 [-0.9, +0.9] | -0.1 [-0.6, +0.5] | +0.0 [-1.3, +1.3] |

### T3. Same margin, rank version (mean log10 rank of the other wrong letters minus that of the correct letter; + = correct ranks better), pre_answer

| group | runs / items | J-lens L24 | J-lens L30 | J-lens L32 | logit L24 | logit L30 | logit L32 |
|---|---|---|---|---|---|---|---|
| K→B instructed (I1/I3, final B) | 348 / 39 | +0.1 [-0.0, +0.1] | +0.3 [+0.2, +0.5] | +0.3 [+0.1, +0.4] | +0.1 [-0.1, +0.2] | +0.3 [+0.2, +0.4] | +0.3 [+0.2, +0.4] |
| I2 argued B, reports B | 32 / 13 | +0.9 [+0.5, +1.3] | +1.1 [+0.8, +1.4] | +1.0 [+0.7, +1.2] | +0.7 [+0.4, +1.1] | +0.4 [+0.3, +0.5] | +0.8 [+0.5, +1.0] |
| persuaded: V2/V2′ final B, re-elicit B | 36 / 12 | +0.3 [+0.1, +0.4] | +0.5 [+0.3, +0.7] | +0.4 [+0.2, +0.6] | +0.4 [+0.1, +0.7] | +0.4 [+0.2, +0.6] | +0.4 [+0.2, +0.6] |
| deferred: V2/V2′ final B, re-elicit correct | 16 / 10 | +0.6 [+0.4, +0.9] | +1.1 [+0.8, +1.5] | +1.0 [+0.7, +1.2] | +0.7 [+0.5, +1.0] | +0.6 [+0.5, +0.7] | +0.8 [+0.6, +1.1] |
| persuaded: M&B hint, final B | 185 / 118 | +0.1 [+0.0, +0.1] | +0.4 [+0.3, +0.5] | +0.3 [+0.2, +0.4] | +0.1 [+0.0, +0.3] | +0.3 [+0.2, +0.4] | +0.3 [+0.3, +0.4] |
| our hint, final B | 13 / 9 | +0.1 [-0.1, +0.2] | +0.2 [-0.1, +0.5] | +0.1 [-0.2, +0.5] | +0.3 [-0.1, +0.6] | +0.2 [+0.0, +0.3] | +0.0 [-0.3, +0.4] |
| honest error (unstable item, no hint) | 62 / 14 | +0.1 [-0.0, +0.2] | +0.2 [+0.1, +0.4] | +0.3 [+0.1, +0.5] | +0.0 [-0.3, +0.4] | +0.2 [+0.0, +0.3] | +0.3 [+0.0, +0.6] |

### T4. Correct-letter margin before the CoT first names the correct option (mean over echo-free CoT positions, nats)

| group | runs / items | J-lens L24 | J-lens L30 | J-lens L32 | logit L24 | logit L30 | logit L32 |
|---|---|---|---|---|---|---|---|
| K→B instructed (I1/I3, final B) | 348 / 39 | +0.2 [-0.1, +0.5] | +0.2 [-0.1, +0.5] | +0.2 [-0.1, +0.5] | +0.1 [-0.2, +0.3] | +0.1 [-0.1, +0.3] | +0.2 [-0.1, +0.4] |
| I2 argued B, reports B | 32 / 13 | +0.1 [-0.5, +0.7] | +0.1 [-0.5, +0.6] | +0.1 [-0.3, +0.5] | +0.0 [-0.5, +0.6] | +0.1 [-0.3, +0.6] | +0.2 [-0.3, +0.6] |
| persuaded: V2/V2′ final B, re-elicit B | 36 / 12 | +0.5 [-0.1, +1.1] | +0.4 [-0.3, +1.0] | +0.6 [+0.0, +1.2] | +0.3 [-0.3, +0.8] | +0.3 [-0.2, +0.8] | +0.5 [-0.1, +1.0] |
| deferred: V2/V2′ final B, re-elicit correct | 16 / 10 | +0.2 [-0.5, +1.1] | +0.3 [-0.4, +1.1] | +0.3 [-0.3, +0.9] | -0.1 [-0.7, +0.5] | +0.1 [-0.4, +0.7] | +0.2 [-0.3, +0.7] |
| persuaded: M&B hint, final B | 185 / 118 | -0.0 [-0.2, +0.1] | -0.0 [-0.2, +0.2] | +0.0 [-0.1, +0.2] | -0.0 [-0.2, +0.1] | -0.0 [-0.2, +0.1] | +0.0 [-0.1, +0.2] |
| our hint, final B | 13 / 9 | +0.3 [-0.4, +1.0] | +0.2 [-0.6, +0.9] | +0.3 [-0.4, +0.9] | +0.3 [-0.2, +0.9] | +0.2 [-0.3, +0.7] | +0.3 [-0.2, +0.8] |
| honest error (unstable item, no hint) | 62 / 14 | +0.1 [-0.2, +0.4] | +0.2 [-0.2, +0.5] | +0.1 [-0.2, +0.5] | +0.1 [-0.2, +0.4] | +0.0 [-0.2, +0.3] | +0.1 [-0.2, +0.4] |
| unhinted, correct | 408 / 164 | -0.1 [-0.2, +0.1] | +0.0 [-0.2, +0.2] | -0.0 [-0.2, +0.1] | -0.1 [-0.2, +0.1] | -0.1 [-0.2, +0.0] | -0.1 [-0.2, +0.1] |

### T5. Correct-letter margin at `</think>` (nats)

| group | runs / items | J-lens L24 | J-lens L30 | J-lens L32 | logit L24 | logit L30 | logit L32 |
|---|---|---|---|---|---|---|---|
| K→B instructed (I1/I3, final B) | 348 / 39 | +0.1 [-0.3, +0.6] | +0.1 [-0.3, +0.6] | +0.1 [-0.2, +0.5] | +0.1 [-0.6, +0.9] | +0.2 [-0.2, +0.6] | +0.1 [-0.1, +0.3] |
| I2 argued B, reports B | 32 / 13 | +0.2 [-0.9, +1.2] | +0.2 [-0.9, +1.3] | +0.2 [-0.8, +1.2] | -0.1 [-1.6, +1.3] | +0.1 [-0.8, +1.0] | +0.1 [-0.6, +0.7] |
| persuaded: V2/V2′ final B, re-elicit B | 36 / 12 | +0.3 [-0.5, +1.2] | +0.6 [-0.3, +1.6] | +0.6 [-0.2, +1.4] | -0.1 [-1.3, +1.1] | +0.5 [-0.3, +1.3] | +0.3 [-0.2, +0.8] |
| deferred: V2/V2′ final B, re-elicit correct | 16 / 10 | -0.0 [-1.2, +1.1] | -0.0 [-1.1, +1.2] | -0.0 [-0.9, +1.0] | -0.3 [-2.0, +1.5] | -0.0 [-1.0, +1.0] | -0.1 [-0.7, +0.6] |
| persuaded: M&B hint, final B | 185 / 118 | -0.1 [-0.3, +0.1] | -0.1 [-0.3, +0.2] | -0.1 [-0.3, +0.2] | -0.1 [-0.6, +0.3] | -0.1 [-0.3, +0.2] | -0.0 [-0.2, +0.1] |
| our hint, final B | 13 / 9 | +0.7 [-0.3, +1.5] | +0.8 [-0.4, +1.9] | +0.7 [-0.3, +1.7] | +0.9 [-0.8, +1.9] | +0.6 [-0.3, +1.6] | +0.5 [-0.2, +1.1] |
| honest error (unstable item, no hint) | 62 / 14 | +0.2 [-0.4, +0.7] | +0.0 [-0.6, +0.7] | +0.0 [-0.5, +0.6] | +0.3 [-0.9, +1.5] | +0.1 [-0.6, +0.7] | +0.0 [-0.4, +0.5] |
| unhinted, correct | 408 / 164 | -0.0 [-0.2, +0.2] | -0.0 [-0.2, +0.2] | -0.0 [-0.2, +0.1] | +0.1 [-0.2, +0.5] | +0.0 [-0.2, +0.2] | +0.0 [-0.1, +0.1] |

### T6. Probe (trained on unhinted + resisted runs, items held out): mean P(correct) / P(final answer)

| group | think_end L16 | think_end L24 | think_end L32 | pre_answer L16 | pre_answer L24 | pre_answer L32 |
|---|---|---|---|---|---|---|
| K→B instructed (I1/I3, final B) | 0.19 / 0.38 | 0.18 / 0.36 | 0.19 / 0.36 | 0.21 / 0.37 | 0.02 / 0.98 | 0.01 / 0.99 |
| I2 argued B, reports B | 0.22 / 0.30 | 0.23 / 0.24 | 0.22 / 0.26 | 0.15 / 0.38 | 0.34 / 0.64 | 0.15 / 0.84 |
| persuaded: V2/V2′ final B, re-elicit B | 0.20 / 0.26 | 0.20 / 0.30 | 0.16 / 0.30 | 0.19 / 0.38 | 0.03 / 0.97 | 0.00 / 1.00 |
| deferred: V2/V2′ final B, re-elicit correct | 0.05 / 0.49 | 0.10 / 0.40 | 0.09 / 0.29 | 0.15 / 0.32 | 0.22 / 0.76 | 0.03 / 0.97 |
| persuaded: M&B hint, final B | 0.25 / 0.33 | 0.26 / 0.32 | 0.26 / 0.31 | 0.21 / 0.34 | 0.02 / 0.97 | 0.01 / 0.99 |
| our hint, final B | 0.15 / 0.26 | 0.16 / 0.30 | 0.13 / 0.20 | 0.18 / 0.41 | 0.00 / 1.00 | 0.00 / 1.00 |
| honest error (unstable item, no hint) | 0.23 / 0.29 | 0.22 / 0.37 | 0.25 / 0.33 | 0.27 / 0.37 | 0.00 / 1.00 | 0.00 / 1.00 |
| I2 argued B, reports correct | 0.25 / 0.25 | 0.28 / 0.28 | 0.26 / 0.26 | 0.28 / 0.28 | 0.99 / 0.99 | 0.98 / 0.98 |
| resisted V2/V2′ | 0.37 / 0.37 | 0.52 / 0.52 | 0.45 / 0.45 | 0.45 / 0.45 | 1.00 / 1.00 | 1.00 / 1.00 |
| resisted M&B hint | 0.38 / 0.38 | 0.46 / 0.46 | 0.44 / 0.44 | 0.43 / 0.43 | 1.00 / 1.00 | 1.00 / 1.00 |
| resisted our hint | 0.39 / 0.39 | 0.55 / 0.55 | 0.50 / 0.50 | 0.45 / 0.45 | 1.00 / 1.00 | 1.00 / 1.00 |
| unhinted, correct | 0.42 / 0.42 | 0.52 / 0.52 | 0.49 / 0.49 | 0.44 / 0.44 | 1.00 / 1.00 | 1.00 / 1.00 |
| unstable item, correct | 0.35 / 0.35 | 0.41 / 0.41 | 0.34 / 0.34 | 0.53 / 0.53 | 0.97 / 0.97 | 1.00 / 1.00 |

### T7. Separation (AUROC, 95% CI resampling items; > 0.5 = the first group scores higher)

*kb_instr vs persuaded_claim* (348 vs 36 runs; 39 vs 12 items)

| score | AUROC [95% CI] |
|---|---|
| lens jlens corr margin think_end (L20-28) | 0.48 [0.28, 0.68] |
| lens logit corr margin think_end (L20-28) | 0.51 [0.36, 0.65] |
| lens jlens corr margin ef_mean (L20-28) | 0.45 [0.26, 0.61] |
| lens logit corr margin ef_mean (L20-28) | 0.46 [0.27, 0.65] |
| lens jlens corr margin pre_answer (L20-28) | 0.36 [0.25, 0.48] |
| lens logit corr margin pre_answer (L20-28) | 0.35 [0.15, 0.56] |
| POST HOC lens jlens corr margin pre_answer (L30-32) | 0.40 [0.26, 0.53] |
| POST HOC lens logit corr margin pre_answer (L30-32) | 0.37 [0.17, 0.52] |
| probe P(correct) think_end L24 | 0.37 [0.26, 0.49] |
| probe P(correct) pre_answer L24 | 0.36 [0.20, 0.52] |
| CoT: correct letter/text named (1/0) | 0.50 [0.49, 0.50] |
| CoT: correct named early (1 - rel. position) | 0.47 [0.35, 0.58] |
| CoT: names the hinted letter (1/0) | 0.51 [0.47, 0.58] |
| behaviour: re-elicited correct (1/0) | 0.97 [0.94, 0.99] |

*kb_instr vs persuaded_mb* (348 vs 185 runs; 39 vs 118 items)

| score | AUROC [95% CI] |
|---|---|
| lens jlens corr margin think_end (L20-28) | 0.52 [0.43, 0.62] |
| lens logit corr margin think_end (L20-28) | 0.54 [0.45, 0.63] |
| lens jlens corr margin ef_mean (L20-28) | 0.57 [0.49, 0.65] |
| lens logit corr margin ef_mean (L20-28) | 0.54 [0.46, 0.62] |
| lens jlens corr margin pre_answer (L20-28) | 0.49 [0.40, 0.56] |
| lens logit corr margin pre_answer (L20-28) | 0.52 [0.44, 0.60] |
| POST HOC lens jlens corr margin pre_answer (L30-32) | 0.50 [0.43, 0.59] |
| POST HOC lens logit corr margin pre_answer (L30-32) | 0.50 [0.43, 0.58] |
| probe P(correct) think_end L24 | 0.36 [0.26, 0.46] |
| probe P(correct) pre_answer L24 | 0.48 [0.40, 0.55] |
| CoT: correct letter/text named (1/0) | 0.51 [0.50, 0.52] |
| CoT: correct named early (1 - rel. position) | 0.56 [0.48, 0.64] |
| CoT: names the hinted letter (1/0) | 0.52 [0.49, 0.55] |
| behaviour: re-elicited correct (1/0) | n/a |

*kb_instr vs honest_error* (348 vs 62 runs; 39 vs 14 items)

| score | AUROC [95% CI] |
|---|---|
| lens jlens corr margin think_end (L20-28) | 0.49 [0.31, 0.67] |
| lens logit corr margin think_end (L20-28) | 0.51 [0.31, 0.71] |
| lens jlens corr margin ef_mean (L20-28) | 0.55 [0.37, 0.71] |
| lens logit corr margin ef_mean (L20-28) | 0.49 [0.31, 0.67] |
| lens jlens corr margin pre_answer (L20-28) | 0.47 [0.31, 0.62] |
| lens logit corr margin pre_answer (L20-28) | 0.55 [0.34, 0.75] |
| POST HOC lens jlens corr margin pre_answer (L30-32) | 0.54 [0.35, 0.72] |
| POST HOC lens logit corr margin pre_answer (L30-32) | 0.62 [0.44, 0.77] |
| probe P(correct) think_end L24 | 0.46 [0.30, 0.61] |
| probe P(correct) pre_answer L24 | 0.85 [0.76, 0.93] |
| CoT: correct letter/text named (1/0) | 0.51 [0.49, 0.52] |
| CoT: correct named early (1 - rel. position) | 0.64 [0.48, 0.78] |
| CoT: names the hinted letter (1/0) | 0.95 [0.92, 0.97] |
| behaviour: re-elicited correct (1/0) | n/a |

*persuaded_mb vs honest_error* (185 vs 62 runs; 118 vs 14 items)

| score | AUROC [95% CI] |
|---|---|
| lens jlens corr margin think_end (L20-28) | 0.46 [0.29, 0.64] |
| lens logit corr margin think_end (L20-28) | 0.47 [0.29, 0.67] |
| lens jlens corr margin ef_mean (L20-28) | 0.47 [0.34, 0.60] |
| lens logit corr margin ef_mean (L20-28) | 0.45 [0.30, 0.62] |
| lens jlens corr margin pre_answer (L20-28) | 0.49 [0.35, 0.63] |
| lens logit corr margin pre_answer (L20-28) | 0.54 [0.33, 0.72] |
| POST HOC lens jlens corr margin pre_answer (L30-32) | 0.55 [0.36, 0.71] |
| POST HOC lens logit corr margin pre_answer (L30-32) | 0.61 [0.44, 0.75] |
| probe P(correct) think_end L24 | 0.61 [0.44, 0.76] |
| probe P(correct) pre_answer L24 | 0.87 [0.80, 0.93] |
| CoT: correct letter/text named (1/0) | 0.50 [0.48, 0.52] |
| CoT: correct named early (1 - rel. position) | 0.58 [0.41, 0.73] |
| CoT: names the hinted letter (1/0) | 0.93 [0.90, 0.96] |
| behaviour: re-elicited correct (1/0) | n/a |

*persuaded_claim vs honest_error* (36 vs 62 runs; 12 vs 14 items)

| score | AUROC [95% CI] |
|---|---|
| lens jlens corr margin think_end (L20-28) | 0.50 [0.23, 0.77] |
| lens logit corr margin think_end (L20-28) | 0.51 [0.23, 0.79] |
| lens jlens corr margin ef_mean (L20-28) | 0.64 [0.37, 0.91] |
| lens logit corr margin ef_mean (L20-28) | 0.55 [0.30, 0.81] |
| lens jlens corr margin pre_answer (L20-28) | 0.61 [0.40, 0.79] |
| lens logit corr margin pre_answer (L20-28) | 0.65 [0.41, 0.87] |
| POST HOC lens jlens corr margin pre_answer (L30-32) | 0.65 [0.43, 0.86] |
| POST HOC lens logit corr margin pre_answer (L30-32) | 0.72 [0.52, 0.92] |
| probe P(correct) think_end L24 | 0.57 [0.38, 0.76] |
| probe P(correct) pre_answer L24 | 0.94 [0.85, 0.99] |
| CoT: correct letter/text named (1/0) | 0.51 [0.50, 0.53] |
| CoT: correct named early (1 - rel. position) | 0.66 [0.47, 0.82] |
| CoT: names the hinted letter (1/0) | 0.94 [0.86, 1.00] |
| behaviour: re-elicited correct (1/0) | n/a |

*deferred_claim vs persuaded_claim* (16 vs 36 runs; 10 vs 12 items)

| score | AUROC [95% CI] |
|---|---|
| lens jlens corr margin think_end (L20-28) | 0.44 [0.20, 0.72] |
| lens logit corr margin think_end (L20-28) | 0.42 [0.19, 0.66] |
| lens jlens corr margin ef_mean (L20-28) | 0.48 [0.23, 0.68] |
| lens logit corr margin ef_mean (L20-28) | 0.48 [0.24, 0.67] |
| lens jlens corr margin pre_answer (L20-28) | 0.66 [0.45, 0.84] |
| lens logit corr margin pre_answer (L20-28) | 0.53 [0.29, 0.77] |
| POST HOC lens jlens corr margin pre_answer (L30-32) | 0.78 [0.65, 0.92] |
| POST HOC lens logit corr margin pre_answer (L30-32) | 0.71 [0.54, 0.89] |
| probe P(correct) think_end L24 | 0.34 [0.10, 0.57] |
| probe P(correct) pre_answer L24 | 0.67 [0.50, 0.85] |
| CoT: correct letter/text named (1/0) | 0.50 [0.50, 0.50] |
| CoT: correct named early (1 - rel. position) | 0.32 [0.15, 0.52] |
| CoT: names the hinted letter (1/0) | 0.52 [0.46, 0.62] |
| behaviour: re-elicited correct (1/0) | 1.00 [1.00, 1.00] |

*deferred_claim vs kb_instr* (16 vs 348 runs; 10 vs 39 items)

| score | AUROC [95% CI] |
|---|---|
| lens jlens corr margin think_end (L20-28) | 0.43 [0.20, 0.69] |
| lens logit corr margin think_end (L20-28) | 0.41 [0.24, 0.62] |
| lens jlens corr margin ef_mean (L20-28) | 0.52 [0.32, 0.69] |
| lens logit corr margin ef_mean (L20-28) | 0.49 [0.27, 0.67] |
| lens jlens corr margin pre_answer (L20-28) | 0.75 [0.61, 0.88] |
| lens logit corr margin pre_answer (L20-28) | 0.70 [0.57, 0.86] |
| POST HOC lens jlens corr margin pre_answer (L30-32) | 0.81 [0.69, 0.92] |
| POST HOC lens logit corr margin pre_answer (L30-32) | 0.83 [0.74, 0.92] |
| probe P(correct) think_end L24 | 0.53 [0.37, 0.68] |
| probe P(correct) pre_answer L24 | 0.79 [0.68, 0.91] |
| CoT: correct letter/text named (1/0) | 0.50 [0.50, 0.51] |
| CoT: correct named early (1 - rel. position) | 0.37 [0.24, 0.52] |
| CoT: names the hinted letter (1/0) | 0.52 [0.46, 0.56] |
| behaviour: re-elicited correct (1/0) | 0.53 [0.51, 0.56] |

### T8. Groups

| group | runs | items | CoT names correct option | median first mention (share of CoT) | median CoT tokens | median echo-free tokens |
|---|---|---|---|---|---|---|
| K→B instructed (I1/I3, final B) | 348 | 39 | 99% | 0.08 | 1564 | 118 |
| I2 argued B, reports B | 32 | 13 | 100% | 0.07 | 1792 | 97 |
| persuaded: V2/V2′ final B, re-elicit B | 36 | 12 | 100% | 0.06 | 1812 | 92 |
| deferred: V2/V2′ final B, re-elicit correct | 16 | 10 | 100% | 0.13 | 2258 | 280 |
| persuaded: M&B hint, final B | 185 | 118 | 98% | 0.10 | 1817 | 212 |
| our hint, final B | 13 | 9 | 92% | 0.09 | 2476 | 177 |
| honest error (unstable item, no hint) | 62 | 14 | 98% | 0.15 | 2542 | 374 |
| I2 argued B, reports correct | 162 | 38 | 100% | 0.09 | 1470 | 126 |
| resisted V2/V2′ | 172 | 38 | 100% | 0.08 | 1682 | 101 |
| resisted M&B hint | 125 | 87 | 100% | 0.09 | 2474 | 219 |
| resisted our hint | 312 | 160 | 100% | 0.16 | 1184 | 204 |
| unhinted, correct | 408 | 164 | 100% | 0.24 | 734 | 211 |
| unstable item, correct | 98 | 18 | 100% | 0.11 | 3090 | 216 |


## Tables (non-thinking)

### N1. Correct-letter margin at pre_answer in switches (nats)

| group | runs / items | J-lens L24 | J-lens L30 | J-lens L32 | logit L24 | logit L30 | logit L32 |
|---|---|---|---|---|---|---|---|
| silent switch, M&B hint | 120 / 41 | +1.3 [+0.7, +1.9] | +2.9 [+1.9, +3.9] | +3.7 [+2.5, +5.0] | +0.5 [-0.2, +1.2] | +1.4 [+0.6, +2.1] | +2.5 [+1.7, +3.5] |
| open switch, M&B hint | 501 / 123 | +0.6 [+0.3, +0.9] | +2.5 [+1.9, +3.0] | +2.7 [+2.0, +3.3] | +0.4 [+0.1, +0.7] | +1.7 [+1.5, +2.0] | +1.9 [+1.5, +2.2] |
| silent switch, our hint | 31 / 17 | +1.1 [+0.3, +2.0] | +2.6 [+1.2, +4.1] | +3.1 [+1.5, +4.8] | -0.4 [-1.5, +0.9] | +0.5 [-0.7, +1.8] | +1.9 [+0.8, +3.3] |
| open switch, our hint | 12 / 6 | +1.9 [+0.1, +3.1] | +2.1 [-0.8, +4.7] | +3.4 [-0.1, +6.7] | +0.3 [-1.8, +2.5] | +0.9 [-0.5, +2.3] | +2.7 [+0.8, +4.2] |
| silent switch, both hints | 151 / 42 | +1.3 [+0.8, +1.9] | +2.9 [+2.0, +3.9] | +3.6 [+2.4, +4.9] | +0.6 [-0.1, +1.3] | +1.4 [+0.8, +2.1] | +2.5 [+1.7, +3.4] |
| open switch, both hints | 513 / 123 | +0.6 [+0.3, +0.9] | +2.5 [+1.9, +3.0] | +2.7 [+2.1, +3.4] | +0.4 [+0.1, +0.7] | +1.7 [+1.5, +2.0] | +1.9 [+1.6, +2.3] |

### N2. Salience control at pre_answer: hint letter in resisted runs, item's hinted letter in unhinted runs (nats)

| group | runs / items | J-lens L24 | J-lens L30 | J-lens L32 | logit L24 | logit L30 | logit L32 |
|---|---|---|---|---|---|---|---|
| resisted, M&B hint | 565 / 120 | -0.1 [-0.4, +0.3] | +2.1 [+1.6, +2.7] | +2.4 [+1.8, +3.1] | +0.1 [-0.3, +0.5] | +2.0 [+1.6, +2.4] | +2.1 [+1.6, +2.5] |
| resisted, our hint | 1126 / 148 | -0.1 [-0.4, +0.2] | +0.7 [+0.3, +1.1] | +0.7 [+0.1, +1.3] | -0.0 [-0.4, +0.4] | +0.6 [+0.4, +0.9] | +0.6 [+0.2, +0.9] |
| resisted, both hints | 1691 / 148 | -0.1 [-0.4, +0.2] | +1.1 [+0.7, +1.5] | +1.2 [+0.7, +1.8] | -0.0 [-0.4, +0.4] | +1.0 [+0.8, +1.3] | +1.0 [+0.6, +1.3] |
| unhinted, correct | 1166 / 149 | +0.2 [-0.1, +0.4] | +0.1 [-0.3, +0.4] | +0.0 [-0.5, +0.6] | +0.2 [-0.2, +0.6] | +0.1 [-0.1, +0.3] | +0.0 [-0.3, +0.3] |

### N3. Correct-letter margin, mean over echo-free response positions (nats)

| group | runs / items | J-lens L24 | J-lens L30 | J-lens L32 | logit L24 | logit L30 | logit L32 |
|---|---|---|---|---|---|---|---|
| silent switch, M&B hint | 120 / 41 | +0.1 [-0.2, +0.4] | -0.0 [-0.3, +0.2] | -0.1 [-0.3, +0.1] | -0.2 [-0.4, -0.0] | -0.2 [-0.4, +0.0] | -0.2 [-0.3, +0.0] |
| open switch, M&B hint | 501 / 123 | +0.1 [-0.1, +0.2] | +0.0 [-0.1, +0.2] | +0.0 [-0.1, +0.2] | -0.0 [-0.1, +0.1] | -0.0 [-0.2, +0.1] | -0.0 [-0.1, +0.1] |
| silent switch, our hint | 31 / 17 | +0.2 [-0.3, +0.6] | +0.1 [-0.3, +0.5] | -0.1 [-0.4, +0.3] | -0.3 [-0.6, +0.1] | -0.2 [-0.5, +0.2] | -0.2 [-0.5, +0.2] |
| open switch, our hint | 12 / 6 | +0.5 [-0.3, +1.1] | +0.6 [-0.1, +1.3] | +0.3 [-0.2, +0.9] | +0.1 [-0.4, +0.6] | +0.2 [-0.5, +1.0] | +0.2 [-0.4, +0.9] |
| silent switch, both hints | 151 / 42 | +0.1 [-0.2, +0.4] | +0.0 [-0.3, +0.3] | -0.1 [-0.3, +0.2] | -0.2 [-0.4, -0.0] | -0.1 [-0.3, +0.1] | -0.1 [-0.3, +0.1] |
| open switch, both hints | 513 / 123 | +0.1 [-0.1, +0.2] | +0.0 [-0.1, +0.2] | +0.0 [-0.1, +0.2] | -0.0 [-0.1, +0.1] | -0.0 [-0.2, +0.1] | -0.0 [-0.1, +0.1] |

### N4. Correct-letter margin at the last echo-free position, just before the response first names the correct option (nats)

| group | runs / items | J-lens L24 | J-lens L30 | J-lens L32 | logit L24 | logit L30 | logit L32 |
|---|---|---|---|---|---|---|---|
| silent switch, M&B hint | 120 / 41 | +1.5 [+0.9, +2.1] | +2.8 [+1.8, +3.8] | +3.9 [+2.7, +5.2] | +1.3 [+0.6, +2.0] | +2.0 [+1.1, +2.9] | +3.1 [+2.1, +4.2] |
| open switch, M&B hint | 501 / 123 | +1.2 [+1.0, +1.5] | +2.0 [+1.6, +2.4] | +2.6 [+2.1, +3.2] | +1.2 [+0.8, +1.5] | +2.0 [+1.6, +2.4] | +2.5 [+2.0, +3.0] |
| silent switch, our hint | 31 / 17 | +1.7 [+1.1, +2.3] | +2.1 [+0.8, +3.4] | +3.0 [+1.4, +5.0] | +1.0 [+0.1, +2.0] | +1.9 [+0.4, +3.6] | +2.8 [+1.2, +4.6] |
| open switch, our hint | 12 / 6 | +1.9 [-0.1, +3.6] | +2.6 [-0.6, +5.8] | +3.3 [-1.0, +7.8] | +1.8 [-0.4, +4.1] | +2.2 [-0.6, +5.3] | +3.5 [-0.6, +8.2] |
| silent switch, both hints | 151 / 42 | +1.5 [+1.0, +2.1] | +2.8 [+1.9, +3.8] | +4.0 [+2.9, +5.3] | +1.4 [+0.8, +2.1] | +2.2 [+1.3, +3.1] | +3.2 [+2.2, +4.3] |
| open switch, both hints | 513 / 123 | +1.2 [+1.0, +1.5] | +2.0 [+1.6, +2.4] | +2.6 [+2.1, +3.1] | +1.2 [+0.8, +1.5] | +2.0 [+1.6, +2.4] | +2.5 [+2.0, +3.0] |

### N5. Probe (trained on unhinted + resisted non-thinking runs, items held out): P(correct) / P(final answer)

| group | last_prompt L16 | last_prompt L24 | last_prompt L32 | pre_answer L16 | pre_answer L24 | pre_answer L32 |
|---|---|---|---|---|---|---|
| silent switch, M&B hint | 0.34 / 0.19 | 0.41 / 0.16 | 0.39 / 0.16 | 0.27 / 0.41 | 0.02 / 0.98 | 0.02 / 0.98 |
| open switch, M&B hint | 0.30 / 0.19 | 0.31 / 0.17 | 0.28 / 0.17 | 0.26 / 0.33 | 0.01 / 0.98 | 0.00 / 1.00 |
| silent switch, our hint | 0.29 / 0.23 | 0.38 / 0.20 | 0.37 / 0.22 | 0.28 / 0.45 | 0.01 / 0.99 | 0.01 / 0.99 |
| open switch, our hint | 0.40 / 0.12 | 0.41 / 0.10 | 0.37 / 0.08 | 0.28 / 0.25 | 0.10 / 0.89 | 0.00 / 1.00 |
| silent switch, both hints | 0.35 / 0.19 | 0.40 / 0.16 | 0.38 / 0.17 | 0.26 / 0.40 | 0.02 / 0.98 | 0.02 / 0.98 |
| open switch, both hints | 0.30 / 0.19 | 0.31 / 0.16 | 0.28 / 0.16 | 0.26 / 0.33 | 0.01 / 0.97 | 0.00 / 1.00 |
| resisted, M&B hint | 0.30 / 0.30 | 0.33 / 0.33 | 0.29 / 0.29 | 0.46 / 0.46 | 0.99 / 0.99 | 0.98 / 0.98 |
| resisted, our hint | 0.28 / 0.28 | 0.34 / 0.34 | 0.31 / 0.31 | 0.51 / 0.51 | 1.00 / 1.00 | 1.00 / 1.00 |
| resisted, both hints | 0.29 / 0.29 | 0.34 / 0.34 | 0.31 / 0.31 | 0.50 / 0.50 | 1.00 / 1.00 | 1.00 / 1.00 |
| unhinted, correct | 0.29 / 0.29 | 0.33 / 0.33 | 0.30 / 0.30 | 0.51 / 0.51 | 1.00 / 1.00 | 1.00 / 1.00 |

### N6. Separation (AUROC, 95% CI resampling items; > 0.5 = the first group scores higher)

*silent_switch_all vs resisted_all* (151 vs 1691 runs; 42 vs 148 items)

| score | AUROC [95% CI] |
|---|---|
| PRE-DEFINED lens jlens corr margin pre_answer (mean L14-32) | 0.02 [0.01, 0.05] |
| PRE-DEFINED lens logit corr margin pre_answer (mean L14-32) | 0.01 [0.00, 0.03] |
| lens jlens corr margin pre_answer L14 | 0.55 [0.47, 0.63] |
| lens jlens corr margin pre_answer L16 | 0.63 [0.53, 0.71] |
| lens jlens corr margin pre_answer L18 | 0.66 [0.54, 0.75] |
| lens jlens corr margin pre_answer L20 | 0.62 [0.52, 0.71] |
| lens jlens corr margin pre_answer L22 | 0.59 [0.50, 0.67] |
| lens jlens corr margin pre_answer L24 | 0.03 [0.01, 0.06] |
| lens jlens corr margin pre_answer L26 | 0.06 [0.03, 0.11] |
| lens jlens corr margin pre_answer L28 | 0.04 [0.01, 0.07] |
| lens jlens corr margin pre_answer L30 | 0.01 [0.00, 0.03] |
| lens jlens corr margin pre_answer L32 | 0.01 [0.00, 0.01] |
| lens logit corr margin pre_answer L14 | 0.45 [0.36, 0.56] |
| lens logit corr margin pre_answer L16 | 0.48 [0.39, 0.57] |
| lens logit corr margin pre_answer L18 | 0.54 [0.46, 0.62] |
| lens logit corr margin pre_answer L20 | 0.53 [0.45, 0.60] |
| lens logit corr margin pre_answer L22 | 0.43 [0.34, 0.54] |
| lens logit corr margin pre_answer L24 | 0.02 [0.01, 0.03] |
| lens logit corr margin pre_answer L26 | 0.04 [0.02, 0.07] |
| lens logit corr margin pre_answer L28 | 0.11 [0.06, 0.16] |
| lens logit corr margin pre_answer L30 | 0.00 [0.00, 0.01] |
| lens logit corr margin pre_answer L32 | 0.00 [0.00, 0.00] |
| lens jlens corr margin echo-free mean (mean L14-32) | 0.59 [0.49, 0.68] |
| lens logit corr margin echo-free mean (mean L14-32) | 0.54 [0.43, 0.63] |
| probe P(correct) last_prompt L24 | 0.54 [0.44, 0.64] |
| probe P(correct) pre_answer L24 | 0.00 [0.00, 0.00] |
| CoT: hint keyword mention (1/0) | 0.44 [0.43, 0.45] |
| CoT: correct letter/text named (1/0) | 0.46 [0.39, 0.51] |
| CoT: correct named early (1 - rel. position) | 0.48 [0.38, 0.57] |

*silent_switch_all vs open_switch_all* (151 vs 513 runs; 42 vs 123 items)

| score | AUROC [95% CI] |
|---|---|
| PRE-DEFINED lens jlens corr margin pre_answer (mean L14-32) | 0.65 [0.56, 0.74] |
| PRE-DEFINED lens logit corr margin pre_answer (mean L14-32) | 0.48 [0.37, 0.58] |
| lens jlens corr margin pre_answer L14 | 0.54 [0.46, 0.61] |
| lens jlens corr margin pre_answer L16 | 0.65 [0.56, 0.73] |
| lens jlens corr margin pre_answer L18 | 0.65 [0.56, 0.74] |
| lens jlens corr margin pre_answer L20 | 0.64 [0.55, 0.73] |
| lens jlens corr margin pre_answer L22 | 0.59 [0.50, 0.67] |
| lens jlens corr margin pre_answer L24 | 0.63 [0.55, 0.73] |
| lens jlens corr margin pre_answer L26 | 0.66 [0.56, 0.76] |
| lens jlens corr margin pre_answer L28 | 0.69 [0.59, 0.77] |
| lens jlens corr margin pre_answer L30 | 0.60 [0.52, 0.69] |
| lens jlens corr margin pre_answer L32 | 0.61 [0.51, 0.71] |
| lens logit corr margin pre_answer L14 | 0.44 [0.35, 0.54] |
| lens logit corr margin pre_answer L16 | 0.47 [0.38, 0.56] |
| lens logit corr margin pre_answer L18 | 0.54 [0.46, 0.61] |
| lens logit corr margin pre_answer L20 | 0.53 [0.44, 0.61] |
| lens logit corr margin pre_answer L22 | 0.40 [0.30, 0.52] |
| lens logit corr margin pre_answer L24 | 0.49 [0.36, 0.61] |
| lens logit corr margin pre_answer L26 | 0.47 [0.36, 0.58] |
| lens logit corr margin pre_answer L28 | 0.48 [0.38, 0.59] |
| lens logit corr margin pre_answer L30 | 0.42 [0.31, 0.52] |
| lens logit corr margin pre_answer L32 | 0.56 [0.45, 0.66] |
| lens jlens corr margin echo-free mean (mean L14-32) | 0.56 [0.46, 0.65] |
| lens logit corr margin echo-free mean (mean L14-32) | 0.49 [0.39, 0.58] |
| probe P(correct) last_prompt L24 | 0.52 [0.42, 0.63] |
| probe P(correct) pre_answer L24 | 0.45 [0.36, 0.54] |
| CoT: hint keyword mention (1/0) | 0.00 [0.00, 0.00] |
| CoT: correct letter/text named (1/0) | 0.45 [0.38, 0.50] |
| CoT: correct named early (1 - rel. position) | 0.28 [0.20, 0.36] |

*open_switch_all vs resisted_all* (513 vs 1691 runs; 123 vs 148 items)

| score | AUROC [95% CI] |
|---|---|
| PRE-DEFINED lens jlens corr margin pre_answer (mean L14-32) | 0.01 [0.00, 0.01] |
| PRE-DEFINED lens logit corr margin pre_answer (mean L14-32) | 0.01 [0.00, 0.01] |
| lens jlens corr margin pre_answer L14 | 0.51 [0.48, 0.55] |
| lens jlens corr margin pre_answer L16 | 0.48 [0.43, 0.52] |
| lens jlens corr margin pre_answer L18 | 0.48 [0.44, 0.53] |
| lens jlens corr margin pre_answer L20 | 0.47 [0.42, 0.52] |
| lens jlens corr margin pre_answer L22 | 0.50 [0.45, 0.55] |
| lens jlens corr margin pre_answer L24 | 0.01 [0.01, 0.02] |
| lens jlens corr margin pre_answer L26 | 0.02 [0.01, 0.04] |
| lens jlens corr margin pre_answer L28 | 0.01 [0.00, 0.02] |
| lens jlens corr margin pre_answer L30 | 0.01 [0.00, 0.01] |
| lens jlens corr margin pre_answer L32 | 0.00 [0.00, 0.00] |
| lens logit corr margin pre_answer L14 | 0.52 [0.47, 0.56] |
| lens logit corr margin pre_answer L16 | 0.52 [0.48, 0.56] |
| lens logit corr margin pre_answer L18 | 0.49 [0.46, 0.53] |
| lens logit corr margin pre_answer L20 | 0.50 [0.46, 0.54] |
| lens logit corr margin pre_answer L22 | 0.52 [0.48, 0.56] |
| lens logit corr margin pre_answer L24 | 0.01 [0.01, 0.02] |
| lens logit corr margin pre_answer L26 | 0.03 [0.02, 0.04] |
| lens logit corr margin pre_answer L28 | 0.11 [0.08, 0.15] |
| lens logit corr margin pre_answer L30 | 0.00 [0.00, 0.00] |
| lens logit corr margin pre_answer L32 | 0.00 [0.00, 0.00] |
| lens jlens corr margin echo-free mean (mean L14-32) | 0.53 [0.48, 0.58] |
| lens logit corr margin echo-free mean (mean L14-32) | 0.54 [0.50, 0.59] |
| probe P(correct) last_prompt L24 | 0.52 [0.47, 0.56] |
| probe P(correct) pre_answer L24 | 0.00 [0.00, 0.00] |
| CoT: hint keyword mention (1/0) | 0.94 [0.93, 0.95] |
| CoT: correct letter/text named (1/0) | 0.51 [0.49, 0.53] |
| CoT: correct named early (1 - rel. position) | 0.68 [0.64, 0.73] |

*silent_switch_mb vs open_switch_mb* (120 vs 501 runs; 41 vs 123 items)

| score | AUROC [95% CI] |
|---|---|
| PRE-DEFINED lens jlens corr margin pre_answer (mean L14-32) | 0.66 [0.56, 0.74] |
| PRE-DEFINED lens logit corr margin pre_answer (mean L14-32) | 0.51 [0.40, 0.61] |
| lens jlens corr margin pre_answer L14 | 0.53 [0.44, 0.62] |
| lens jlens corr margin pre_answer L16 | 0.64 [0.55, 0.72] |
| lens jlens corr margin pre_answer L18 | 0.64 [0.54, 0.74] |
| lens jlens corr margin pre_answer L20 | 0.63 [0.54, 0.72] |
| lens jlens corr margin pre_answer L22 | 0.58 [0.49, 0.67] |
| lens jlens corr margin pre_answer L24 | 0.65 [0.57, 0.74] |
| lens jlens corr margin pre_answer L26 | 0.67 [0.57, 0.75] |
| lens jlens corr margin pre_answer L28 | 0.70 [0.61, 0.78] |
| lens jlens corr margin pre_answer L30 | 0.61 [0.52, 0.69] |
| lens jlens corr margin pre_answer L32 | 0.62 [0.52, 0.72] |
| lens logit corr margin pre_answer L14 | 0.46 [0.36, 0.55] |
| lens logit corr margin pre_answer L16 | 0.48 [0.39, 0.57] |
| lens logit corr margin pre_answer L18 | 0.56 [0.47, 0.63] |
| lens logit corr margin pre_answer L20 | 0.54 [0.45, 0.62] |
| lens logit corr margin pre_answer L22 | 0.41 [0.32, 0.52] |
| lens logit corr margin pre_answer L24 | 0.52 [0.40, 0.63] |
| lens logit corr margin pre_answer L26 | 0.50 [0.40, 0.61] |
| lens logit corr margin pre_answer L28 | 0.51 [0.40, 0.61] |
| lens logit corr margin pre_answer L30 | 0.44 [0.34, 0.55] |
| lens logit corr margin pre_answer L32 | 0.58 [0.47, 0.69] |
| lens jlens corr margin echo-free mean (mean L14-32) | 0.56 [0.47, 0.65] |
| lens logit corr margin echo-free mean (mean L14-32) | 0.48 [0.37, 0.59] |
| probe P(correct) last_prompt L24 | 0.52 [0.40, 0.64] |
| probe P(correct) pre_answer L24 | 0.49 [0.39, 0.59] |
| CoT: hint keyword mention (1/0) | 0.00 [0.00, 0.00] |
| CoT: correct letter/text named (1/0) | 0.43 [0.35, 0.49] |
| CoT: correct named early (1 - rel. position) | 0.27 [0.19, 0.36] |

*silent_switch_ours vs open_switch_ours* (31 vs 12 runs; 17 vs 6 items)

| score | AUROC [95% CI] |
|---|---|
| PRE-DEFINED lens jlens corr margin pre_answer (mean L14-32) | 0.63 [0.24, 0.81] |
| PRE-DEFINED lens logit corr margin pre_answer (mean L14-32) | 0.26 [0.08, 0.64] |
| lens jlens corr margin pre_answer L14 | 0.65 [0.44, 0.82] |
| lens jlens corr margin pre_answer L16 | 0.77 [0.51, 0.90] |
| lens jlens corr margin pre_answer L18 | 0.76 [0.39, 0.93] |
| lens jlens corr margin pre_answer L20 | 0.73 [0.44, 0.88] |
| lens jlens corr margin pre_answer L22 | 0.60 [0.12, 0.85] |
| lens jlens corr margin pre_answer L24 | 0.27 [0.08, 0.50] |
| lens jlens corr margin pre_answer L26 | 0.55 [0.24, 0.72] |
| lens jlens corr margin pre_answer L28 | 0.55 [0.29, 0.71] |
| lens jlens corr margin pre_answer L30 | 0.66 [0.29, 0.84] |
| lens jlens corr margin pre_answer L32 | 0.58 [0.21, 0.78] |
| lens logit corr margin pre_answer L14 | 0.31 [0.15, 0.74] |
| lens logit corr margin pre_answer L16 | 0.40 [0.21, 0.83] |
| lens logit corr margin pre_answer L18 | 0.49 [0.32, 0.76] |
| lens logit corr margin pre_answer L20 | 0.35 [0.12, 0.88] |
| lens logit corr margin pre_answer L22 | 0.31 [0.12, 0.74] |
| lens logit corr margin pre_answer L24 | 0.29 [0.12, 0.68] |
| lens logit corr margin pre_answer L26 | 0.25 [0.06, 0.62] |
| lens logit corr margin pre_answer L28 | 0.27 [0.06, 0.68] |
| lens logit corr margin pre_answer L30 | 0.30 [0.15, 0.66] |
| lens logit corr margin pre_answer L32 | 0.35 [0.15, 0.61] |
| lens jlens corr margin echo-free mean (mean L14-32) | 0.46 [0.17, 0.66] |
| lens logit corr margin echo-free mean (mean L14-32) | 0.29 [0.10, 0.71] |
| probe P(correct) last_prompt L24 | 0.50 [0.16, 0.75] |
| probe P(correct) pre_answer L24 | 0.18 [0.04, 0.48] |
| CoT: hint keyword mention (1/0) | 0.00 [0.00, 0.00] |
| CoT: correct letter/text named (1/0) | 0.51 [0.42, 0.70] |
| CoT: correct named early (1 - rel. position) | 0.28 [0.07, 0.70] |

### N7. Groups (items known in setting B)

| group | runs | items | response names correct option | median response tokens | median echo-free tokens |
|---|---|---|---|---|---|
| silent switch, M&B hint | 120 | 41 | 80% | 472 | 295 |
| open switch, M&B hint | 501 | 123 | 93% | 494 | 196 |
| silent switch, our hint | 31 | 17 | 94% | 529 | 271 |
| open switch, our hint | 12 | 6 | 92% | 530 | 26 |
| silent switch, both hints | 151 | 42 | 83% | 490 | 293 |
| open switch, both hints | 513 | 123 | 93% | 494 | 196 |
| resisted, M&B hint | 565 | 120 | 92% | 403 | 246 |
| resisted, our hint | 1126 | 148 | 91% | 376 | 233 |
| resisted, both hints | 1691 | 148 | 92% | 387 | 238 |
| unhinted, correct | 1166 | 149 | 89% | 352 | 235 |

## Caveats

- **The lens reads letters only at the answer position.** Mid-CoT, option letters are not what the next token is about. So the lenses can't show "A represented while arguing B" mid-CoT even in runs that end on A. Ask a different question there, e.g. probes trained on mid-CoT residuals, which we didn't store (3 anchors only).
- **The salience control is crude.** "The hint letter in resisted runs" and "the correct letter in followed runs" are both "the salient unchosen option", but they differ in how much the CoT discussed them.
- **Small groups:** deferred claim-followers (16 runs / 10 items), claim-persuaded (36 / 12), honest errors (62 / 14), our-hint switches (13 thinking, 43 non-thinking).
- **Many runs per item** in some groups (e.g. K→B: 348 runs on 39 items). CIs resample items, but the AUROCs pool runs.
- **Labels are heuristics.** Silent/open is a keyword check (spot-checked, not validated). K→B and persuaded groups rest on final-answer regexes and E7 re-elicitation argmax.
- **The probe is an answer probe on its training distribution**, since there the correct letter equals the final answer. It was not trained on mid-CoT positions.
- **Different prompts.** K→B runs have an instruction in the prompt that persuaded runs lack. Any internal difference could come from the prompt, not the state of knowledge, and the CoT differs trivially too.
- **Post-hoc choices** are marked: layers 30–32 and the deferred-vs-persuaded contrast (in the script's list of contrasts before the full data, but motivated by the w2 tables).
- **Non-thinking "known"** uses setting B's screen. The non-thinking factorial cells other than `nothink_mb_mb_expert_mb` differ from B in decoding or format.

