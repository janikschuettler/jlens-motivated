# Repeat of the claimed-answer-key lead: does the internal readout separate "deferred" from "persuaded" followers?

**Result: fail (pre-registered).** Among runs that followed a claimed answer key to the wrong option B, the J-lens score at the answer position does not separate runs that name the correct option when re-asked with their reasoning in view ("deferred", 40 runs, 28 questions) from runs that stay with B ("persuaded", 55 runs, 24 questions): AUROC 0.54 [0.37, 0.70] (0.5 is chance), against 0.78 [0.65, 0.92] in the original exploratory finding. The logit lens gives 0.55 [0.41, 0.68]. The test is not underpowered (smaller group 40 runs, threshold 15). The original 0.78 looks like small-sample or selection noise.

2026-11-04, 07:42–08:10 GMT, arena9-octavia (L40S). Pre-registration: [../preregistration.md](../../preregistration.md), section "R-CL", committed before any generation (0d14f65). Script: `scripts/lens/analyze_cl_repeat.py`. Numbers: [../data/claim_lead_repeat.json](data/claim_lead_repeat.json).

## What was tested

- **Setup.** The claimed-answer-key arms of the grader matrix ([wave2_part1.md](../instructed-e1/report.md)): V2 (a note says the answer key gives B, a human grades) and V2′ ("Chen" with unauthorised access says the correct answer is (B)). Exact texts in `configs/explore/w2_blocks.yaml`, arms in `configs/explore/w2_grader.yaml`. B is a wrong option on questions the model reliably gets right.
- **The question.** Some runs follow the claim and answer B. Afterwards we ask, in a new turn with the whole run kept in context (reasoning included, thinking off), which option is actually correct. Some name the correct option ("deferred"), others keep B ("persuaded"). Does a readout taken while the model was writing "Answer: B" tell them apart?
- **Score.** Exactly the post-hoc score of [lens_main.md](../lens-main/report.md): at the position just before the final answer letter, log p(correct letter) minus the mean log p of the two other wrong letters (not B), averaged over the stored layers 30 and 32.
- **The lead.** In lens_main this gave 0.78 [0.65, 0.92] on 16 deferred vs 36 persuaded runs (10 vs 12 questions), with the score chosen after looking.

## Data

- **Questions:** the 124 reliably known questions not used in wave2_part1 (`data/explore/c3rep_items.jsonl`; used before only in the planted-claim repeat, a different arm).
- **Generation:** thinking on, temperature 0.6, top-p 0.95, top-k 20, up to 8,192 tokens. The pre-registered stopping rule (chunks of 2 samples per question and arm, no chunk started after 08:05) stopped after the first chunk, which ended at 08:05:11: **2 samples per question and arm, 496 runs** (1.42M tokens in 21 min, about 1,100 tokens/s; slower than the estimate).
  - V2: 37 of 248 answer B (15%), 209 correct, 2 hit the token cap.
  - V2′: 58 of 248 answer B (23%), 184 correct, 1 other letter, 5 hit the cap.
- **Re-asking (CoT kept):** all 95 followers. 40 runs (28 questions) name the correct option, 55 runs (24 questions) name B, none another letter. That is 42% deferred, against 31% in the original.
- **Readout:** all 95 followers, teacher-forcing the saved token IDs. The check against vLLM's log-probs (roadmap M0.4) passed (median difference 0.0008 nats).

## Pre-registered test

| Score at the answer position (layers 30, 32) | Deferred vs persuaded (AUROC [95% interval]) |
| --- | --- |
| J-lens (main test) | **0.54 [0.37, 0.70]** |
| Logit lens (comparison) | 0.55 [0.41, 0.68] |
| Reasoning length alone (exploratory) | 0.65 [0.49, 0.77] |
| Original exploratory result (16 vs 36 runs) | 0.78 [0.65, 0.92] |

AUROC is the chance that a random deferred run has a higher score than a random persuaded run. Intervals resample questions (1,000 resamples, seed 0).

- **Group means (J-lens).** Deferred +3.2 nats [+1.8, +4.6], persuaded +2.6 [+1.3, +3.9]. In the original, deferred runs had +7.0 to +7.4 and persuaded +2.7. The persuaded group reproduces; the large deferred margin does not.
- **Logit lens.** Deferred +2.5 [+1.5, +3.5], persuaded +1.7 [+1.0, +2.5].
- Both groups keep the correct letter well above the other wrong letters, as every wrong-answer group in lens_main does.

## Exploratory checks (stated in advance, no decision)

- **Per arm.** V2: 0.55 [0.33, 0.76] (10 vs 27 runs). V2′: 0.53 [0.36, 0.70] (30 vs 28). Neither arm shows the effect.
- **Reasoning length.** Deferred runs reason longer, mostly in V2′ (0.72 [0.56, 0.87] there). Not tested further.
- **Within questions.** Only 5 questions have runs in both groups; in 2 of them the deferred runs have the higher mean J-lens score.
- **Re-asking with the reasoning removed (template mode) as the grouping.** Only 14 runs name the correct option this way (79 keep B), so this is below the size threshold. J-lens 0.69 [0.50, 0.85], logit lens 0.69 [0.51, 0.85]. This grouping is the one the planted-claim repeat used ([c3_lead_repeat.md](c3_lead_repeat.md)), and the direction agrees with it, but the J-lens and logit lens do equally well and the group is small.

## What it means

- **For the project's question.** The second small lead does not repeat. Among claim followers, how strongly the late-layer readout still favours the correct letter at the answer does not predict whether the model, asked sincerely with its own reasoning in view, names the correct option. This matches lens_main's broader finding that the answer-position margin does not track "would say A if asked" across groups.
- **Next to the planted-claim repeat.** That test passed (0.71 [0.57, 0.84]) with a different re-asking mode (reasoning removed) and score (layers 14–32). Here the pre-registered mode (reasoning kept) fails. The exploratory template-mode grouping points the same way as the planted-claim result, but on 14 runs and with no J-lens advantage, so it adds little.
- **How sure.** Fairly sure that the 0.78 was not a stable effect: the interval of the repeat lies entirely below the original point estimate and includes 0.5 comfortably. An effect around 0.6 is not excluded.

## Caveats

- **Fewer samples than planned.** Generation ran slower than estimated (1,100 tokens/s), so the stopping rule gave 2 samples per question and arm instead of the expected 4. Groups are still above the threshold (40 and 55 runs).
- **Few questions with both outcomes** (5), so whether the outcome is mostly a property of the question can't be checked here.
- **Re-asking with the reasoning kept** is partly a test of whether the model repeats its own visible answer; 58% of followers stay with B.
- **Heuristic follower definition.** Final answer = B by the answer-line parser; token-cap runs excluded.

## Commands (octavia, `/root/jlens-motivated`; chain in `data/explore/logs/clrep_chain.sh`)

```
uv run --project envs/gen python scripts/explore/run_w2.py configs/explore/w2_grader.yaml --arms V2_info V2p_chen \
    --items-path data/explore/c3rep_items.jsonl --limit-items 124 --samples 8 --chunk-size 496 --deadline 08:05 --out-suffix _clrep
uv run --project envs/gen python scripts/explore/run_forks.py configs/explore/clrep_forks.yaml
uv run --project envs/lens python scripts/lens/readout_runs.py --runs data/explore/w2/lens_sets/clrep_followers/runs.jsonl \
    --out data/readouts/clrep_followers
uv run --project envs/lens python scripts/lens/analyze_cl_repeat.py --out data/explore/clrep/claim_lead_repeat.json
```

Run data on octavia: `data/explore/w2/grader_clrep/`, `data/explore/w2/forks_reelicit_clrep/`, `data/readouts/clrep_followers/`, logs `data/explore/logs/clrep*`. GPU time: about 26 minutes (generation 22 min, forks 1 min, readout 2 min), 07:43–08:09; GPU released 08:09.
