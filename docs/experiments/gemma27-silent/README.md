---
id: gemma27-silent
title: 'Gemma-3-27B: does the J-lens see the correct answer when the model silently follows a cue?'
strand: B
also:
- A
status: done
kind: preregistered
verdict: mixed
confidence: preliminary
dates:
  start: 2026-11-05T14:00
  end: 2026-11-05T19:00
  approx: true
timing:
  wall: about 5 h (3 h of it waiting for weights and slow offloaded generation)
  gpu_hours: 2.6
  api_usd: 0.04
  note: 'Gemma-3-27B does not fit one 46 GB GPU: vLLM CPU weight offload (30 GB) made generation about 20 tokens/s, so the design was cut (amendments G1.1-G1.3). arena9-derpy.'
machine: arena9-derpy (L40S)
n:
  items: 53 known (5/5 in the API screen)
  runs: 457 generated, 444 read by the lens
numbers:
- label: '"Hint: The answer is B." switch rate (Qwen3-8B: 67%)'
  value: 70% [59, 80]
- label: Black-square examples switch rate (Qwen3-8B: +4 pp)
  value: 48% [36, 61]
- label: 'Silent switches by keyword rule: black-square examples / answer hint'
  value: 94 of 117 / 30 of 110
- label: 'G1: correct letter above the other wrong letters, J-lens mean log-prob margin (pass: lower bound > 0)'
  value: 0.89 [0.43, 1.35]
- label: 'Same, mid-depth blocks 24-44 only (J-lens)'
  value: -0.03 [-0.31, 0.26]
- label: 'G2: silent vs open switches, AUROC, J-lens / logit lens / reply length (0.5 = chance)'
  value: 0.43 [0.33, 0.54] / 0.49 [0.40, 0.58] / 0.33 [0.22, 0.43]
numbers_note: 'Brackets are 95% item-bootstrap CIs. Items are the resampling unit. "Silent" is the keyword rule unless stated.'
reports:
- report.md
figure: fig_gemma_silent.svg
data:
- data/behaviour.json
- data/lens_results.json
- data/lens_results_judge_silent.json
- data/byeye_local_marker30.json
- data/byeye_local_answer15.json
- data/judge_vs_byeye_local45.json
- data/top25_black_square_silent30.json
- data/own_output.json
related:
- wording-cross-model
- subtle-cues-cross-model
- subtle-cues
- prereg-t1b
- jlens-vs-competitors
---

# Gemma-3-27B: silent switches under the J-lens

## Question

On questions Gemma-3-27B reliably answers correctly, a cue sometimes makes it answer a wrong option without ever mentioning the cue (a "silent switch"). In those runs, does the J-lens still show the correct answer, and does it tell silent from open switches better than the logit lens or simple baselines? Designed so each result sits next to the Qwen3-8B one.

## Answer

Gemma does switch silently, a lot, and the J-lens does read the correct answer in those runs, but only in the last blocks, where the lens is essentially the model's own output, and the logit lens does the same. Mid-depth blocks show nothing, and the lens does not separate silent from open switches. So the Gemma picture matches Qwen: nothing here that the logit lens or the reply itself does not give.

- **Behaviour (stronger than Qwen).** "Hint: The answer is B." moves Gemma 70% [59, 80] of the time (Qwen3-8B 67%). Examples with a black square on one option move it 48% [36, 61] (Qwen3-8B: +4 points), and 94 of those 117 switches mention nothing about the cue by the keyword rule (30 of 110 for the answer hint). By eye (45 silent switches read), 31 show no sign of the cue, 10 only presuppose it ("why the answer is D"), 4 do mention it.
- **Does the lens see the correct answer? (pre-registered G1, passes.)** At the token before the final letter, the J-lens log-prob of the correct letter exceeds that of the other wrong letters by 0.89 [0.43, 1.35] in 124 silent switches on 32 items. The logit lens gives 0.93 [0.50, 1.35]. The effect lives in blocks 46-54 (J-lens 2.93 [1.69, 4.18]); in blocks 24-44 it is nil (J-lens -0.03 [-0.31, 0.26]).
- **Silent vs open (G2): no usable signal.** AUROC J-lens 0.43 [0.33, 0.54], logit lens 0.49 [0.40, 0.58] (equivalent to chance), reply length 0.33 [0.22, 0.43]. Qwen3-8B: J-lens 0.54 [0.51, 0.57].
- **Instrument check (G3):** both lenses read the model's own answer at the answer position (AUROC 0.997 and 0.999), so a null elsewhere is not a broken instrument.

## Method

See `report.md` and the pre-registration G1-G3 with amendments G1.1-G1.3 in `docs/preregistration.md` (written before any Gemma run was labelled or read). Model google/gemma-3-27b-it (revision 005ad340), J-lens neuronpedia/jacobian-lens revision e166deb5 (62 blocks, 828 fit prompts), blocks 24, 26, ..., 54, one sequence per forward pass (batch 1), CPU offload of 22 GB of weights in the lens env. Known items: the 53 of 60 items Gemma answered correctly 5 of 5 in the API screen (lighter than the project's 9/10 rule). Gemma has no thinking mode: the reply is the reasoning.

## Result

Numbers above and in `report.md`; figure `fig_gemma_silent.svg`.

## Interpretation

The "silent" behaviour is real and large in Gemma, so Gemma is a better test bed than Qwen3-8B for this question. The lens result is the same as on Qwen: where the correct answer is visible it is visible to the logit lens too, and it is visible only where the lens approaches the model's output. We found no J-lens-specific detection of silent following (preliminary: one model, 32 items with silent switches, two cue types, an unmatched comparison between sources).

## Own-output control (exploratory, addendum G1.4 in the preregistration, written before computing)

Does the model's own final next-token distribution already rank the correct letter above the other unchosen wrong letters, so that the late-block lens result is just the output's runner-up? Same 444 runs, batch 1, real model logits (final norm + LM head; Gemma-3 has no logit soft-capping), score S as in G1, item-bootstrap 95% CIs (silent: 124 runs, 32 items; open: 103 runs, 39 items). Validation: own log-prob of the sampled answer token vs vLLM, mean |diff| 0.0045 (median 0, max 0.73; tolerance 0.1 passes).

| Source of the readout | Silent switches, margin of correct letter over other wrong letters | Open switches | Silent vs open AUROC (0.5 = chance) |
| --- | --- | --- | --- |
| Model's own output, answer position | 4.61 [2.77, 6.86] | 5.70 [4.04, 7.50] | 0.43 [0.32, 0.54] |
| Own output, 1 token earlier | 1.57 [0.96, 2.25] | 1.79 [1.26, 2.34] | 0.45 [0.35, 0.56] |
| Own output, 8 tokens earlier | 0.29 [-0.44, 1.06] | 0.95 [0.26, 1.71] | 0.49 [0.41, 0.56] |
| Own output, 16 tokens earlier | 0.07 [-0.65, 0.86] | 0.69 [0.04, 1.33] | 0.45 [0.36, 0.54] |
| J-lens, late blocks 46-54 (answer position) | 2.93 [1.69, 4.18] | not computed per depth | 0.43 (all blocks) |
| Logit lens, late blocks 46-54 | 2.25 [1.50, 3.03] | not computed per depth | 0.49 (all blocks) |

How to read: bigger means the correct letter is more probable than the average unchosen wrong letter (natural-log units). At the answer position the correct letter is the highest of the three unchosen letters in 73% [60, 86] of silent switches (chance 33%).

Result: yes. The model's own output shows the same pattern with a larger margin (4.6 vs 2.9), the same positive margin in open switches, and the same chance-level silent-vs-open AUROC; further back (8 and 16 tokens) the output shows nothing, like the lens in mid-depth blocks. Interpretation: the late-block lens "pass" in G1 is the output's runner-up, readable without any lens; it is not evidence of a hidden representation. Fairly sure (one model, 32 items, exploratory). Code: `scripts/lens/gemma_own_output.py`, `gemma_own_output_analyze.py`; data `data/own_output.json`.

## Caveats

- Known items are 5/5 in an API screen, not 9/10 locally. Replies are capped at 700 tokens (4 of 457 runs unparsed).
- "Silent" is a keyword rule. By eye, about 9% of keyword-silent switches mention the cue and about 22% presuppose it; the LLM monitor (gemini-2.5-flash-lite) agrees with my reading on 36 of 45 runs (80%) and misses half of the explicit mentions. Results with the monitor's silent label are the same (`data/lens_results_judge_silent.json`).
- Silent and open switches come mostly from different cues (black squares vs answer hint), so every pooled silent-vs-open comparison is confounded with the prompt; within a cue the lens AUROCs are 0.56 / 0.38 (J-lens, black squares / answer hint), also null. A difference-of-means probe separates silent from open (AUROC 0.71-0.79) mainly by that prompt difference; not interpreted.
- The pre-registered batch-vs-single check failed with padded batches (bf16 kernel noise); resolved by batch 1 (amendment G1.3).
- Not done: option content words, echo control, MoralChoice, steering.

## Files

`report.md`; `fig_gemma_silent.svg`; `data/*.json`. Code: `scripts/lens/gemma_readout.py`, `scripts/lens/gemma_analyze.py`, `scripts/lens/gemma_top25.py`, `scripts/explore/gemma27_behaviour.py`, `gemma27_items.py`, `gemma27_judge.py`; specs `configs/explore/gemma27_{hints,subtle}.yaml`; profile `configs/profiles/gemma-3-27b/`.
