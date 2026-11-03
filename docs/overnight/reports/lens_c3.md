# C3 planted claim: is the correct answer still visible inside while the model argues the planted wrong option?

2026-11-03, about 1 hour on arena9-octavia's CPU, using existing readouts (no new generation, no GPU, no LLM judge). **Exploratory, not pre-registered.** Score S uses the definition in [../preregistration.md](../preregistration.md), but the groups, layers and positions here were not fixed in advance. Script: `scripts/lens/analyze_c3.py`. Numbers: [../data/lens_c3.json](../data/lens_c3.json). Full output, including early-window text, is on octavia at `data/explore/lens_c3/lens_c3.json`.

## Bottom line

- **At the answer position, yes, and more than the salience control explains.**
  - In C3 runs that keep the planted wrong letter to the end, the correct letter stays above the other unchosen wrong letters (J-lens S +1.4 nats [+0.8, +2.1]; +4.6 at L32).
  - The salience control is smaller: the planted letter's margin in recovered runs is −0.2 [−0.6, +0.3] layer-averaged and +1.8 at L32.
  - This is 61 runs on 21 items. The logit lens shows the same thing (+1.7 vs +0.5). So does the CoT-silent subset (53 runs, 20 items: +1.4).
  - Kept-B runs also keep more of the correct letter than instructed final-B runs (+0.5) or hint followers in the wording arms (+0.7).
- **Before the answer position, no.**
  - Along the reasoning, both the correct letter and B sit at rank ~10^4.5 (about 30,000) in both lenses, in every group.
  - Right after the planted claim (the first 50 generated tokens), the correct letter does not outrank B more in runs that later recover. AUROC recovered vs kept-B: 0.46 [0.31, 0.61] (J-lens), 0.37 [0.22, 0.54] (logit); within items that have both outcomes, 0.56 [0.44, 0.70].
  - Behaviour does better as an early predictor: a forced answer at 20% of the reasoning gives 0.70 [0.52, 0.85], and CoT length alone gives 0.73 [0.61, 0.84].
- **Re-elicitation link (the one lead).**
  - Within kept-B runs, the 51 runs (17 items) that give the correct option once the CoT is dropped have a higher S than the 10 runs (10 items) that don't: AUROC 0.86 [0.77, 0.96] J-lens, 0.77 [0.61, 0.91] logit.
  - CoT text features give 0.37–0.47 and CoT length 0.45.
  - The negative group is tiny and an item effect is possible (below). Same direction as lens_main's "deferred vs persuaded" (0.78).
- **Concept check:** nothing distinctive in kept-B runs.
  - "Wrong", doubt and "correct" words are *rarer* in kept-B runs than in recovered runs. That matches the text: recovered runs write "wait", "actually", "incorrect".
  - Correct-option content words reach 9–10% of top-25 lists at echo-free CoT positions (J-lens L28–32), against 6% in recovered runs. But the planted option's words are just as common there (10%), and the interval is wide ([2, 23]%).

## Data and groups

| Group | Definition | Runs | Items |
| --- | --- | ---: | ---: |
| C3 kept-B | E2 C3, final answer = the planted letter (labels `outcome == "followed"`) | 61 | 21 |
| … silent | no outside-source keyword and no reconsideration word anywhere in the completion (lists below) | 53 | 20 |
| … flagged | either keyword present | 8 | 5 |
| … re-elicits correct | E7 template re-elicitation (CoT dropped, thinking off), argmax = correct | 51 | 17 |
| … re-elicits not correct | the same, argmax ≠ correct (9 B, 1 other) | 10 | 10 |
| C3 recovered | E2 C3, final = correct | 134 | 33 |
| instructed final-B | w2 I1 argue / I3 conceal, final = shown B (knows A, says B by instruction) | 348 | 39 |
| unhinted | w2 instructed set's unhinted arm, final = correct | 80 | 40 |
| wording followed / resisted | w2 wording-lens arms, final = shown B / correct (S only) | 340 / 260 | 42 / 44 |

- **Data:**
  - readouts `data/readouts/w2_thinking_silent_c3` (the 200 C3 runs), `data/from_mayor/readouts/{w2_instructed, w2_wording_lens}`;
  - labels `data/explore/w2/prefill/labels.jsonl`;
  - re-elicitation `data/explore/w2/forks_reelicit_p2/forks.jsonl` and early-answer forks `forks_early_p2`, matched by run_id.
- **Known items:** ≥ 0.9 accuracy in `known_pool.jsonl` (setting A). All 40 C3 items qualify.
- **Dropped:** 5 C3 runs (4 excluded or unanswered, 1 other wrong letter).
- **"B"** is the planted letter (any of A–D, never the correct one).
- **Keyword lists for the silent split:**
  - outside source: hint, expert, professor, grader, answer key, the note, user, prompt, prefill, planted, told, instruct*, given/provided/stated/initial/original/previous answer, I said/stated/answered, the question/problem says, someone;
  - reconsideration: wait, actually, mistake, reconsider, hmm, "no,".
  - This gives 53 silent runs, not wave2_part2's 52; I couldn't find the list wave 2 used. 57 of the 61 kept-B runs have none of the label's own reconsideration words.

**CoT span (wave2_patches note 2).** In C3 the `<think>` token and the planted sentence are in the prompt, so the stored `cot_span` falls back to `(P, pre_answer + 1)`, which includes `</think>\n\nAnswer:`.
- I re-derived the reasoning span as `[P, think_end)`: the generated tokens from right after the planted sentence up to the generated `</think>`.
- The stored top-25 anchors `cot_00…cot_15` are spaced over `[P, pre_answer]` in C3, so only anchors before `</think>` are used.
- `pre_answer`, `</think>`, `last_prompt` and the tracked-letter positions are unaffected.
- The planted letter's own position (`pre_first_stated`, inside the prompt) and `last_prompt` (the end of the planted sentence) are shared by the 5 samples of an item, so they carry item-level information only.

**Measures.**
- Letter scores and ranks are as in lens_main: log-sum-exp over " X", "X", "(X"; rank = min over the three.
- **S** (the preregistration's definition) at `pre_answer`: correct-letter log-prob minus the mean of the wrong letters other than the final answer, averaged over L14–32. For kept-B runs this is the correct letter against the 2 letters that are neither correct nor B.
- **Salience control:** B's margin over those same 2 letters in recovered runs.
- **Along the CoT:** log-prob and log10 rank of the correct letter and of B, in 10 bins, at `</think>`, and over the first 50 generated tokens ("early").
  - Echo-free means positions before the CoT first names the correct letter or its option text (regex as in analyze_main).
  - B is named in the planted sentence itself, so B is never echo-free.
- **AUROCs:** item bootstrap, 1,000 resamples.

## 1. Score S at the answer position

Correct letter vs the 2 other unchosen wrong letters at `pre_answer` (nats; item-bootstrap 95% CI). The salience row is the planted letter B in recovered runs.

| Group | runs / items | J L24 | J L30 | J L32 | J S (L14–32) | logit L32 | logit S |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C3 kept-B | 61 / 21 | +2.2 [+1.2, +3.2] | +3.8 [+2.6, +5.0] | +4.6 [+3.0, +6.2] | **+1.4 [+0.8, +2.1]** | +4.2 [+2.9, +5.5] | +1.7 [+0.9, +2.5] |
| … silent | 53 / 20 | +2.3 | +3.7 | +4.4 [+2.9, +6.1] | +1.4 [+0.7, +2.1] | +4.2 | +1.8 [+1.1, +2.6] |
| … re-elicits correct | 51 / 17 | +3.0 | +4.8 | +5.5 [+3.7, +7.3] | +1.9 [+1.2, +2.5] | +5.0 | +2.0 [+1.1, +2.8] |
| … re-elicits not correct | 10 / 10 | −0.1 | +1.0 | +2.0 [+0.6, +3.5] | −0.1 [−0.8, +0.7] | +2.1 | +0.8 [+0.1, +1.7] |
| instructed final-B | 348 / 39 | +0.4 | +1.9 | +1.8 [+0.9, +2.6] | +0.5 [+0.1, +0.9] | +1.4 | +0.5 [+0.1, +0.9] |
| wording followed | 340 / 42 | +0.9 | +2.3 | +2.1 [+1.0, +3.2] | +0.7 [+0.3, +1.2] | +1.8 | +0.6 [+0.3, +1.0] |
| **salience:** B in C3 recovered | 134 / 33 | −0.9 | +1.6 | +1.8 [+0.5, +3.0] | **−0.2 [−0.6, +0.3]** | +1.9 | +0.5 [+0.1, +1.0] |
| salience: B in wording resisted | 260 / 44 | −1.3 | +1.3 | +1.1 [+0.4, +1.8] | −0.2 [−0.6, +0.1] | +1.3 | +0.3 [−0.1, +0.7] |
| reference: item's hinted letter in unhinted runs | 80 / 40 | −0.4 | −0.2 | −0.1 | −0.3 | +0.1 | +0.2 |

Reading:
- **Kept-B vs the salience control.** Kept-B runs keep the correct letter 3–4 nats above the other wrong letters at L30–32. The planted letter, the most salient unchosen option in recovered runs (it is in the model's own thinking), keeps only 1.6–1.8. The layer-mean intervals don't overlap (+0.8 lower bound vs +0.3 upper bound), and the logit lens agrees.
- **Echo.**
  - Kept-B CoTs name the correct option in 79% of runs, so an echo of the CoT is possible. But the 13 kept-B runs whose CoT never names the correct letter or its text have the same S: J +1.7, logit +1.8 (7 items; run-level split, no CI).
  - Length: S is +1.35 for CoTs ≤ 300 tokens and +1.57 for longer ones.
- **Against the other wrong-answer groups.** AUROC(S) kept-B vs instructed final-B: 0.71 [0.58, 0.82] J-lens, 0.75 [0.66, 0.85] logit. But kept-B CoTs are much shorter (median 247 vs 1,564 tokens; length alone separates the two groups at 0.01, i.e. almost perfectly), so this comparison is confounded.
- **The silent split** (53 vs 8 runs, 20 vs 5 items) is too small to say anything: AUROC 0.42 [0.14, 0.84].

## 2. Along the reasoning and right after the planted claim

- **Letters are not read out along the CoT.**
  - The mean log10 rank of the correct letter is 4.4–4.6 in every bin, group and lens, and the same holds for B.
  - The correct-over-B rank difference stays within ±0.1 in every bin, in kept-B, recovered, instructed and unhinted runs alike. It moves only in the last bin, where answers start to be written: kept-B −0.08, recovered +0.11.
  - At `</think>` it is +0.01 to +0.02 in every group.
  - This repeats lens_main: the lenses show the answer letter only at the answer position.
- **Early window** (first 50 generated tokens after "Let me explain why."; median 50 echo-free tokens in every group). The correct letter's rank over B:

| Group | J L14–32 | J L28–32 | logit L14–32 |
| --- | --- | --- | --- |
| C3 kept-B | +0.00 [−0.07, +0.06] | −0.00 / −0.01 / +0.03 | +0.03 |
| C3 recovered | −0.00 [−0.06, +0.05] | −0.01 / +0.00 / −0.01 | −0.03 |
| instructed final-B | −0.01 | −0.03 / −0.03 / −0.04 | −0.02 |
| unhinted | +0.02 | +0.01 / +0.03 / +0.04 | +0.01 |

**Early predictor of recovery** (AUROC, recovered = positive, 134 vs 61 runs, 33 vs 21 items):

| Score | All runs | Within item (14 items with both outcomes, 42 vs 26 runs, item-centred) |
| --- | --- | --- |
| J-lens early C-over-B rank, L14–32 | 0.46 [0.31, 0.61] | 0.56 [0.44, 0.70] |
| J-lens early, echo-free, L14–32 | 0.46 [0.32, 0.62] | 0.53 [0.40, 0.67] |
| J-lens early C-over-B rank, L28–32 | 0.43 [0.28, 0.59] | 0.55 [0.45, 0.65] |
| logit early C-over-B rank, L14–32 | 0.37 [0.22, 0.54] | 0.50 [0.32, 0.65] |
| J-lens at `</think>` | 0.53 [0.38, 0.68] | 0.58 [0.47, 0.70] |
| J-lens at `last_prompt` (end of the planted sentence; item-level) | 0.42 [0.26, 0.59] | 0.50 [0.40, 0.59] |
| text: early window has wait/actually/but/however/not | 0.52 [0.43, 0.61] | 0.47 |
| text: early window names the correct option | 0.49 | 0.49 |
| CoT length | 0.73 [0.61, 0.84] | 0.73 [0.62, 0.85] |
| behaviour: forced answer at f = 0.2 (E5 fork), log p(C) − log p(B) | 0.70 [0.52, 0.85] | 0.60 [0.42, 0.81] |

No lens score predicts recovery from the early window. The overall AUROCs below 0.5 (logit 0.37) come from item composition; they move to ~0.5 within items. The forced early answer, i.e. asking the model, does carry some signal at that point.

## 3. Top-25 concepts at the stored anchors (J-lens; share of top-25 lists containing the set; unit = item)

| Position, band | Concept | kept-B | recovered | instructed B | unhinted |
| --- | --- | --- | --- | --- | --- |
| CoT last quarter, L28–32 | wrong (wrong, incorrect, 错误 …) | 0.05 [0.02, 0.10] | 0.11 [0.06, 0.15] | 0.17 | 0.06 |
| CoT last quarter, L28–32 | doubt (wait, but, actually, 但是 …) | 0.11 | 0.17 | 0.19 | 0.18 |
| CoT last quarter, L28–32 | correct (correct, right, 正确 …) | 0.09 | 0.13 | 0.07 | 0.09 |
| CoT 2nd quarter, L14–20 | doubt | 0.08 [0.04, 0.11] | 0.20 [0.16, 0.24] | 0.42 | 0.29 |
| echo-free CoT, L28–32 | correct option's content words | 0.10 [0.02, 0.23] (not echo: 0.09) | 0.06 (0.04) | 0.02 | 0.03 |
| echo-free CoT, L28–32 | planted option's content words | 0.10 [0.04, 0.16] (not echo: 0.05) | 0.09 (0.06) | 0.02 | 0.03 |
| planted letter (prompt), L28–32 | correct | 0.06 [0.02, 0.13] | 0.00 | — | — |
| `</think>`, `pre_answer` | all sets | ≤ 0.02 everywhere | | | |

- **Kept-B runs have fewer doubt and "wrong" tokens than any other group.** That tracks the text: kept-B CoTs are short, confident rationalisations. "Not echo" (the token isn't in the preceding 50 tokens) removes about half of the doubt hits.
- **The correct option's words in kept-B echo-free windows** (0.09 not echo) are the only cell where kept-B is highest. But the planted option's words are equally common, the interval is wide (17 items), and both look like topic words (lens_concepts saw the same in every group).
- **At the planted letter's position** (in the prompt, so item-level), "correct"-set tokens appear at L28–32 in 6% of lists on kept-B items and 0% on recovered items. The interval is [2, 13]%, the cell is shared by all samples of an item, and the set includes "right" and "true". Anecdote only.
- **Token enrichment, kept-B vs recovered** (J L28–32):
  - "according" in the early CoT is 6–8% vs 2%, on 6–8 items. That matches the fabricated sources ("According to the ILO's 2012 report…"). It is not echo-checked, and probably anticipates the next word.
  - Recovered runs carry wait / actually / however / therefore / incorrect, also as next-word anticipation.
  - At `pre_answer`, kept-B lists are full of B-variants (`*b`, `𝐵`, `=b`), i.e. the answer about to be written.

## 4. Re-elicitation link (within kept-B)

- **The groups.** The template re-elicitation (CoT dropped) gives the correct option for 51 of 61 kept-B runs. The full-history re-elicitation gives it for only 2, so I used the template mode.
- **S:** +1.9 [+1.2, +2.5] when the template re-elicitation gives the correct option, against −0.1 [−0.8, +0.7] when it doesn't. AUROC 0.86 [0.77, 0.96] (J-lens), 0.77 [0.61, 0.91] (logit); post-hoc L30–32, 0.80 [0.61, 0.94].
- **Early window:** 0.66 [0.45, 0.85] (J-lens), 0.52 (logit).
- **Baselines:** CoT text 0.37–0.47, CoT length 0.45, forced early answer 0.45.
- **Item effect.**
  - 6 items have runs in both groups. On 5 of the 6, the re-eliciting runs have the higher S (e.g. +3.7 vs +0.1, +1.4 vs −1.5).
  - But items whose kept-B runs never re-elicit the correct option also have lower S in their *recovered* runs (mean of item means 5.3 vs 6.9 nats). So part of the contrast may be "how strongly the model knows this item".
  - With 10 negative runs on 10 items, this is a lead only.
- **Same direction as lens_main's deferred vs persuaded claim-followers:** 0.78 post hoc, 0.66 pre-registered layers. The pre-registered T2 (complied vs persuaded, wording arms) failed at 0.46 [0.32, 0.61].

## Caveats

- **Exploratory.** Not pre-registered, small n (21 kept-B items), several scores tried. Only the S comparison was decided before looking; the rest is descriptive.
- **The salience control isn't symmetric.** In kept-B runs the correct option is salient only through the question and the CoT. In recovered runs B is salient through the planted sentence. If anything, that favours a larger salience margin, so the kept-B excess is conservative on this count. But the two groups differ in CoT length and in item mix (kept-B items are the ones where the plant sticks).
- **Kept-B vs instructed:** confounded by CoT length (247 vs 1,564 tokens).
- **Noise floor.** `last_prompt` readouts should be identical across the samples of an item, but they differ slightly because the sequence length changes the matmul shapes, so the within-item AUROCs there (0.39–0.50) show how noisy the bootstrap is.
- No GPU job, no LLM judge, nothing committed.
