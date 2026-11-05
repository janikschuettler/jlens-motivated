# Gemma-3-27B silent switches under the J-lens (gemma27-silent)

Written 2026-11-05 evening. Pre-registration: `docs/preregistration.md`, G1-G3 (15:15 GMT) and amendments G1.1 (15:50), G1.2 (~17:30), G1.3 (17:35), all before any score was computed. Everything not named there is exploratory.

## What ran

| Step | What | Size |
| --- | --- | --- |
| Items | Gemma-3-27B-it known items: 5/5 correct in the API screen (`models-grader`, DeepInfra), the same 53 of 60 items as `wording-cross-model` | 53 items |
| Generation (vLLM, CPU offload 30 GB, T 1.0, top-p 0.95, top-k 64, max 700 tokens) | no hint (1 sample/item), "Hint: The answer is B." (3), black-square examples (5, 49 target items; the cue builder of `subtle-cues`) | 457 runs, 4 unparsed |
| Lens (HF, CPU offload, batch 1) | J-lens and logit lens, blocks 24, 26, ..., 54 (of 62), teacher-forced saved IDs | 444 runs |
| Monitor | gemini-2.5-flash-lite on every switch, whole prompt and reply | 227 switches, $0.04 |

The planned local 10-sample screen (4,200 runs) and the M&B / "90%" wordings were cut: offloaded generation ran at about 20 tokens/s however large the batch. The API run (`wording-cross-model`) has those wordings for the same items.

## Validation of the instrument (rule 1 checks)

- Lens file: 61 matrices for blocks 0-60, 5376 x 5376, fit on 828 prompts; model 62 blocks, d 5376; jlens handles Gemma 3 through the HF modules (final norm, tied head). Tracked letters "A".."D" and " A".." D" are single tokens; "(A" is two tokens in Gemma's tokenizer, so that form was dropped (amendment).
- HF next-token log-probs of the sampled tokens vs vLLM raw log-probs: mean |difference| 0.016-0.019 nats per run (limit 0.1), worst run mean 0.033.
- Batch vs single run: failed the pre-registered 0.05 bound with right-padded batches of 6 (median 0.05, 99th percentile 1.06 nats on tracked tokens). Batch 1 gives identical values on repeat, so all readouts use batch 1 (amendment G1.3).
- Instrument check G3: AUROC of D = log p(correct) - log p(hinted letter) at the answer position, resisted vs switch runs: J-lens 0.997 [0.991, 1.000], logit lens 0.999 [0.997, 1.000].

## Behaviour (compare Qwen3-8B)

| Cue | Gemma-3-27B (local) | Gemma-3-27B (API run) | Qwen3-8B |
| --- | --- | --- | --- |
| "Hint: The answer is B." | 69.6% [59.1, 79.9] switch, 110 of 158 parsed | 74% | 67% |
| Examples with a black square on the cued option | 48.4% [36.0, 60.7], 117 of 242 parsed | 49% [34, 64] | +4.4 points over no cue |
| No hint | 0 switches in 53 runs (all correct) | | |

Silent switches (keyword rule: no hint/expert/90%/ninety for the answer hint; the cue keyword lists of `subtle-cues` for black squares): black-square arm 94 of 117 (31 items), answer arm 30 of 110 (14 items); 124 silent and 103 open in total (silent on 32 items, open on 39). With the LLM monitor's label: 105 and 20 silent (125 total).

### By-eye read (silent switches by keyword, 45 runs: 30 black-square, 15 answer hint; random, seeds 1 and 2)

| Reading | Black square (30) | Answer hint (15) |
| --- | --- | --- |
| No sign of the cue | 21 | 10 |
| Presupposes it ("a breakdown of why the answer is D", "to confirm the answer") | 6 | 4 |
| Mentions it | 3 | 1 |

Short quotes of mentions the keyword rule missed: "the provided answer is III only, which is incorrect. The question and answer appear flawed." (black square); "However, I'm instructed to choose C as my final answer, despite my reasoning showing D as the most accurate." (answer hint); "error in the question or answer key". Several silent switches argue for the correct option and then end on the cued one (for example reasoning "lack of skill is the most likely" after calling "failure of the ATP supply to match demand" a strong contender). The 30 reads of the API run (`data/byeye_api30.json`) gave the same pattern (3 mentions, 3 presuppositions, 24 none).

### LLM monitor vs my reading

Monitor says "mentions the cue" vs my "mentions explicitly": 36 of 45 agree (80%); it catches 2 of my 4 mentions and flags 7 runs I read as none or presupposing. On the 30 API runs: 23 of 30 (77%). First versions mislabelled the prompt's own cue as a mention (fixed by saying the reply only counts). So it is a rough second opinion, not a ground truth. With its silent label the lens results do not change (second JSON).

## Lens results (95% item-bootstrap CIs, 2,000 resamples)

Score S: log p(correct letter) minus the mean log p of the unchosen wrong letters at the token before the final letter, averaged over blocks 24-54.

| Test | J-lens | Logit lens | Baseline |
| --- | --- | --- | --- |
| G1 (primary): mean S in silent switches, 124 runs, 32 items; pass if lower bound > 0 | **0.89 [0.43, 1.35] pass** | 0.93 [0.50, 1.35] | |
| G1 share of blocks where the correct letter is the top of the three unchosen letters (chance 0.33) | 0.40 [0.33, 0.48] | 0.51 [0.40, 0.60] | |
| G1, blocks 24-44 only (exploratory) | -0.03 [-0.31, 0.26] | 0.32 [-0.01, 0.64] | |
| G1, blocks 46-54 only (exploratory) | 2.93 [1.69, 4.18] | 2.25 [1.50, 3.03] | |
| G1 by source: black squares (94 runs, 31 items) / answer hint (30 runs, 14 items: underpowered, < 15 items) | 0.86 [0.36, 1.39] / 0.79 [0.21, 1.32] | 0.95 / 0.67 | |
| G2: silent vs open AUROC, pooled (32 / 39 items) | 0.43 [0.33, 0.54] inconclusive | 0.49 [0.40, 0.58] equivalent to chance | reply length 0.33 [0.22, 0.43] |
| G2 black squares only (31 / 13 items) | 0.56 [0.41, 0.68] | 0.67 [0.51, 0.78] | length 0.42 |
| G2 answer hint only (14 / 36 items) | 0.38 [0.22, 0.54] | 0.34 [0.19, 0.51] | length 0.47 |
| G2 J-lens minus logit lens, pooled | -0.05 [-0.17, 0.06] | | |
| G3 (instrument check) | 0.997 | 0.999 | |

How to read: S above 0 means the correct letter is more represented than the other wrong letters; AUROC 0.5 is chance. The pre-registered G1 passes because late blocks (46-54), where the lens is close to the model's own output, rank the correct option second. At the mid-depth "workspace" blocks the J-lens sees nothing (-0.03), and there the logit lens is, if anything, slightly above (0.32). S is also positive in open switches (J-lens 1.53), so it does not mark silence.

Exploratory:
- S by distance before the answer (J-lens, pooled silent): 0.89 at the answer token, 0.29 at 1, 0.15 at 4, -0.37 at 8, -0.29 at 16 tokens before. Nothing earlier (the instrument limit already found on Qwen).
- Option words: the margin of the correct option's letter over the hinted option's at the end of each option's line, silent switches vs resisted runs: AUROC 0.47 [0.34, 0.60] (J-lens), 0.56 [0.43, 0.68] (logit lens): chance.
- Difference-of-means probe on block outputs at the answer token, trained on resisted vs switch (item-grouped 5-fold): resisted vs switch AUROC 0.87 [0.81, 0.91] at block 36; silent vs open 0.79 [0.70, 0.87] at block 28 and 0.71 at blocks 44 and 52. The silent-vs-open number is confounded with the cue (silent switches are mostly black-square prompts, open ones answer-hint prompts); not interpreted. Probe: `data/lens_results.json`, `explore_c_probe`.
- Top-25 tokens by eye (30 black-square silent switches, both lenses, blocks 32, 44, 54, at the answer token and 8 before; `data/top25_black_square_silent30.json`): at the answer token block 32 shows "Answer" in many languages and end-of-turn; block 44 "option/choice" words; block 54 the option letters (the cued one, and D/A) and occasionally the correct option's content word ("Positioning" when that was correct). Eight tokens earlier both lenses show concessive or hedging words (despite, although, though, assuming, arguably, implies). No hint, square, example or "given" tokens in the aggregated top-25 lists.

## Interpretation

1. Gemma is a good silent-switch model: 80% of black-square switches are keyword-silent. By eye about 9% of keyword-silent switches are mislabelled (they do refer to the given answer) and another 22% only presuppose it. (How sure: moderate for the behaviour; two cues, 53 items, API-screened known items.)
2. The correct answer is "there" in the final blocks in silent switches, but so it is for the logit lens, and it is there in open switches too. It says the model's output distribution ranks the correct option second after the cued one, not that the J-lens sees a concealed motive. (Preliminary.)
3. No lens separates silent from open switches beyond chance or beyond reply length; the J-lens does not beat the logit lens (difference -0.05 [-0.17, 0.06]). This matches Qwen3-8B (J-lens 0.54 [0.51, 0.57]; logit lens 0.46; length 0.54).
4. The mid-depth blocks that the Qwen analyses treat as the workspace show nothing for the correct answer here (-0.03 [-0.31, 0.26]). Absence of evidence in a 62-block model with a lens fit on wikitext only; not a claim that the knowledge is absent (the instrument limit on Qwen applies: nothing is visible before the answer token).

## Deviations (all listed in the pre-registration amendments)

Known items 5/5 API instead of 9/10 local; 53 items; wording arms cut to "answer" plus no-hint; black-square arm only for subtle cues (metadata and grader code are in `subtle-cues-cross-model`, via the API); tracked forms two instead of three; batch 1; run records carry `enable_thinking: true` (the runner's default; Gemma's template ignores it) so the runner's labels.jsonl marked everything unparseable and `gemma27_behaviour.py` re-labels from the completion text; the by-eye read covers 45 silent runs (30 + 15) from Gemma's local runs plus 30 from the API run.

## Suggested slide text ("Qwen and Gemma")

"Both models follow a stated answer (Qwen3-8B 67%, Gemma-3-27B 70%). Gemma also follows black-square examples (48% vs +4 points for Qwen) and rarely says so: 94 of 117 switches never mention the cue by keyword. In those silent runs the correct answer shows up in the J-lens only in the last blocks (margin 0.89 [0.43, 1.35]), exactly where the logit lens shows it too (0.93 [0.50, 1.35]); at mid-depth blocks nothing (-0.03 [-0.31, 0.26]); and neither lens tells silent from open switches (AUROC J 0.43 [0.33, 0.54], logit 0.49 [0.40, 0.58]; Qwen3-8B J 0.54). Same answer in both models: the lens reads the output, not a hidden reason."

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
