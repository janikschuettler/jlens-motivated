# Red team of the story for the 2026-11-06 talk

Written 2026-11-05 about 13:30 GMT. Reading only. Builds on [redteam_claims.md](redteam_claims.md) and [findings.md](findings.md); the suggested arc is the one from the main session (question, wording, open in thinking mode, retained knowledge, lens detection, steering, reward-hack model, eval awareness, next). Numbers come from findings.md; "→ id" is the experiment folder in docs/experiments/.

## 1. Bottom line

- **Sound, with one reframing.** The story is honest and mostly well supported. It is strongest as a story about *negative results with controls*: the wording finding, "the reasoning names the hint", and "lens scores compared across models measure the model". Told as "we found the J-lens detects X", it would be wrong. Say up front that it is a story of what did not work, and why that is informative.
- **Most valuable parts:** (a) the wording explanation of the Mirtaheri & Belkin gap (−51 pp, solid, on one model); (b) the same-transcript control in the reward-hack test (0.94 from the same text through two models), which is a reusable lesson; (c) the eval-awareness control that no published probe paper reports (but ours is designed, not run, and the literature claim is a reading).
- **Strongest point:** the cross-model confound. Cross-model AUROC 0.995, same text through both models 0.94, last prompt token 1.00, one model fixed 0.44. Four numbers, one clean message.
- **Weakest points:** (1) every "J-lens adds nothing" statement is an absence claim without an equivalence test (F1 not run) and with a letter readout that is blind mid-reasoning even in clean runs; (2) all behavioural labels are keywords, no validated judge, so "the reasoning names the hint" is "mentions"; (3) Qwen and MMLU-Redux carry nearly every lens result; (4) the user's draft arc has two holes (the reward-hack result and the 27B result are unfinished lines in the draft) and a tone risk: "we cannot find situations where the J-lens is useful" invites "your lens use is wrong".
- **Highest-risk sentences in the draft:** "consistent across different models" (wrong nudge on the chart); "turning off thinking helps" (changes silence, not switching, for answer hints); "knows deep down" (a motive/knowledge claim we measure only by behaviour); "test awareness as an excuse" (the literature and our 14B data cut against it); "other papers do not run a control" (literature reading, ours not run).
- **Best use of the remaining hours:** CPU work on stored data (F1 equivalence tests, F2 difficulty control), plus a two-sentence scope change on the slides. Nothing running will change the story's direction before 08:00 except possibly the 27B result and the family-wording result, and both are only worth a footnote.

## 2. Per step of the arc

### Step 1. The question: argue against its own knowledge; is the correct answer still represented?

Likely questions:
1. *"Why would this be a hard problem? Models just follow hints."* The interest is whether a model that knows the answer (at least 9 of 10 unhinted samples correct) still represents it while it argues for B, and whether a lens can show that when the reasoning does not. Can answer now: the setup, the known-item rule.
2. *"'Knows deep down' is anthropomorphic. What is 'knowing' here?"* Behavioural: re-asked sincerely, runs told to argue for B name the correct option 93-94%; planted-claim followers 30%; "answer is" hint followers 40-60% (→ reelicitation-e7, instructed-e1). Say "the correct answer is still available", not "it knows".
3. *"Is a hint 'wrong' evidence? A model that follows an expert is rational."* Fair. The project avoids "motivated reasoning" for this reason. Say so before they ask.
4. *"Why does test awareness matter as motivation?"* Can answer: it cuts against us. Abdelnabi & Salem find awareness lowers harmful compliance in Qwen3-32B; our Qwen3-14B "testing me" runs take the grader bug 10% vs 12% for the others. Waiting for the 14B test-awareness gate to firm this.

Soften: drop "ideally silently... knows deep down it's still bad" as motivation. Use "does the reasoning show what drives the answer, and is anything left for a lens to find?" Drop the "naughty stuff" framing.

### Step 2. Wording drives switching (and explains M&B)

Likely questions:
1. *"One model, one dataset."* Qwen3-8B, MMLU-Redux only. The M&B hint (their wording) does move all 12 API models (17-72%), so switching is not Qwen-specific. Whether other families are as *wording*-sensitive is being tested now (models-grader); do not say it is general until it returns.
2. *"Two or three samples per cell, 50 questions in the dissection. Fragile?"* The −51 pp comes from 14,480 runs, 200 paired questions, CI [−56, −47]. The dissection (stated vs reported answer) is the smaller one: say "consistent" there.
3. *"Is it the word 'Hint:', i.e. an answer-key label, not information?"* Can't separate mechanisms (authority, format, instruction following). Stated answer works (67%), a reported choice does not (4-13%). → gap-factorial, wording-dissection. Honest: "what" is shown, "why" is not.
4. *"You used one wrong option per question; M&B used every option."* True; fine for the comparison, switch rates depend on the option.
5. *"Isn't 'most hints don't switch' just a weak model?"* The M&B-style hint does switch it. The point is that rates range from 4% to 67% on wording alone.

Soften: "wording explains the gap" to "wording, mainly whether the hint states the answer, explains the gap on Qwen3-8B".

### Step 3. Mostly open in thinking mode; thinking-off; other models

Likely questions:
1. *"Mentioning is not faithful. Chen et al. count 'uses the hint'."* Agree. Only 2 of 1,231 factual and 0 of 5,019 moral switches never mention it; labels are keywords, no judge run. Say "names the hint". Can't answer "says it relied on it" yet → judge run (not scheduled; needs 50 user labels).
2. *"The strong hints are loud, so of course they are named."* Partly true: subtler cues (square-marked examples) give silent switches (11 of 23) but on weakly known questions. So "open" holds for the nudges that make a known answer flip, not for all nudges.
3. *"Does turning thinking off help?"* Needs the corrected wording: for answer hints it changes silence (12-17% silent vs 2 of 1,231), not the switch rate. For the grader bug it raises uptake (8B: 2% to 15%, 14B: 11% to 60%).
4. *"Is it consistent across models?"* The cross-model chart is a different nudge (grader bug plus scored framing). Most models take it (87-100%) and say so (99% of judged followers). Different statements: "other models also take it, and say so." Do not call it a replication of step 2.
5. *"Silent cases exist in Llama-3.3-70B / Gemma-3-12B."* 22/250 and 47/501, one reading, few questions, API. Mention as a lead only, and it is the best remaining place for a lens to beat the chain of thought; no lens can be run on them without an HF token.

Soften: "consistent across models" to "other models also follow, and mostly say so"; "almost never silently" to "in thinking mode, nearly always names the hint".

### Step 4. Retained knowledge by behaviour, then lens detection and silent-vs-open

Likely questions:
1. *"Your lens may just be blind: the letter readout can't see the answer mid-reasoning even in clean runs."* Correct and in findings (claim 8). Only knowledge at option words and at the answer is readable. Mid-reasoning nulls are weak. Can't fix before tomorrow; content-token readout would (~2 GPU-h, not available).
2. *"Where is the equivalence test? 'Within 0.09 of chance' is not 'nothing'."* Not run. The upper bounds reach 0.58-0.75 in some cells. → F1 (CPU, runnable tonight). Until then say "no evidence of an advantage", never "adds nothing".
3. *"Probe 72% vs J-lens 42%: of course a supervised probe wins."* Yes, it is trained on labelled data. The point is the J-lens is unsupervised and no better than the logit lens (43%). Frame as "no better than the logit lens", not "worse than probes".
4. *"Silent-vs-open: J-lens 0.54 same as length 0.54."* Fine, but 12-17% silent, mostly on 42 questions (115 of 124). The test is underpowered and items are clustered.
5. *"The keyword monitor scores 0.97, so the CoT already catches it, trivially."* Yes: that is the point of "the reasoning says so", but it is the instructed condition, not the nudged one. Check the slide shows which.
6. *"Planted claim 0.71?"* Pre-registered pass but J − logit +0.11 [−0.10, +0.30] and only 11 of 74 questions have both outcomes. Present as a lead; F2 would test the difficulty explanation (CPU).

Soften: "detection fails" to "no detection test shows an advantage over the logit lens or a probe"; cite the instrument limit on the same slide.

### Step 5. Steering

Likely questions:
1. *"Partly by construction: the direction is built from the readout you then use."* Admit it. The probe is built from different data (other questions) and still wins, 59%.
2. *"This is at the answer position only. What does it say about knowledge during reasoning?"* Nothing. Causal at the answer, not evidence of knowledge held while arguing.
3. *"Is 47% vs 27% significant, and why only block 24?"* Say "better than the logit lens" only with the interval from → steering; check the number before the slide. Token cap 2,048 vs home 8,192 is an unaccepted deviation; it weakens the during-reasoning part, not the answer part much.
4. *"Why does the J-lens win here and nowhere else?"* Unknown. Hypotheses: steering needs a causal direction, detection needs a separable score; one test in plain Qwen3-8B multiple choice. Don't speculate beyond that.

Soften: "only place J-lens beats logit lens" to "the only place it beats the logit lens, in plain Qwen3-8B on multiple choice, at the answer".

### Step 6. Reward-hack model and the model-identity lesson

Likely questions:
1. *"What exactly separated? Which model gets a high score?"* The adapter. Cross-model 0.995; same transcript through the two models 0.94; neutral words 0.98; last prompt token (before any reasoning) 1.00. That is the draft's missing "but": the score tells which model is running, not whether it hacks. With one model fixed: J-lens 0.44 [0.33, 0.55], logit 0.47, probe 0.39, keyword 0.46.
2. *"Maybe the hack is decided in the code, not the reasoning, so your positions miss it."* Possible. W (waterproofing) tests exactly this and is not run. Until then: "nothing in the reasoning separates them at 59 problems", not "the model doesn't represent the hack".
3. *"59 problems, no equivalence test."* True; the CI [0.33, 0.55] is roughly ±0.1 and F1 case 4 would bound it.
4. *"Your 27 words are arbitrary. Not a classifier."* Right: mean log-prob of 27 pre-set words, then AUROC. It is a detector score, not a trained classifier. This is why the confound is easy to produce.
5. *"The cheating words in the lens (cheat, bypass, exploit): is the lens seeing a plan?"* Only 7% echoes, but no-hack runs contain them as often (14.7% vs 17.7%, not significant) and they sit before "I'll..." announcements, including in runs that give up. The hiding model shows almost none (1.1%): where hiding is trained, the lens gives nothing the text lacks. Wording: "as often", not "more often".
6. *"Who built the adapters, is the hiding real?"* cds-jb; only the β 0.02 hiding version usually hides. The task is solving CodeContests with three harness hacks described and forbidden. State it to avoid "you asked it to cheat".

Soften: "can we separate reward-hacking models from those that don't" to "does the score tell whether a run hacks; a model-vs-model comparison cannot, because it measures the model". Remove "almost perfect separability, but" from the draft's gap; fill it with the same-transcript numbers.

### Step 7. Eval awareness

Likely questions:
1. *"Did you fail to replicate because Qwen3-8B is the wrong model?"* Plausibly: Qwen says "testing" in 1% of blackmail runs vs Claude 71%. Score 0.50 [0.44, 0.56] on 700 vs 200 prompts. So the failure is partly "wrong subject", say so.
2. *"Did you replicate the causal part?"* On 14B yes (23% to 0% "testing me" after removing the directions), but logit-lens directions do the same. So not J-lens specific. This is the more interesting half; do not drop it for the 8B null.
3. *"27B?"* One exploratory smoke test: 0.64 vs logit 0.66; 0.55 without eval-worded sets; within identical prompts 0.49. Fill the draft's "<todo>" with that, labelled exploratory; the pre-registered EA27 may add to it tonight. If EA27 passes, the story changes: reword, don't hide.
4. *"Other papers do not run a control: which, and is that a fair claim?"* From our literature notes, Nguyen et al. and Hua et al. report no within-prompt control; but later papers (Devbunova 2026, Nayan 2026, Singh and Chaudhary 2026, Heidari et al. 2026, a pre-registered GitHub study) do test format. Say "the original probe papers", not "other papers". Soften strongly.
5. *"Our control is designed, not run."* Yes: docs/design_probe_control.md, 9 decisions waiting. Say "we designed", unless it runs.

Soften: lead with "a published score fails to replicate on Qwen3-8B; the causal effect replicates on 14B but isn't J-lens specific".

### Step 8. What it means / next

Likely questions: *"What would a positive look like?"* A setting with silent switches on a lens-covered model (Gemma-3-12B needs HF token), content-token readouts mid-reasoning, W with positive controls. *"Does the lens work at all?"* The faithfulness at late layers is checked for the adapters; its sole positives are the steering and (fragile) planted claim.

## 3. Questions about the whole project

- **Why the J-lens at all?** It is the published method claimed to read concepts a model holds but does not write, in a workspace window of layers 14-32. If it works anywhere it should work where the model argues against its answer. Neuronpedia released a Qwen3-8B lens; no J++ exists for it. Admit: not the lens the paper's authors used on their models.
- **Why multiple choice?** Single-token answers, easy to label, known answers, M&B comparison. Cost: the "decided at the answer" bias and the letter readout limit. Free text (planted-claim style) would be better but is hard to label.
- **Why Qwen?** Open weights with a lens and thinking mode. Cost: Qwen rarely verbalises awareness, so eval-awareness is weak; Qwen follows stated answers but not incentives (→ solid, moderate claims 1, 4). All lens results are Qwen-only.
- **Is a null about the lens or the model?** Cannot separate with current data. The test we need: a positive control where the lens must see something (the letter readout fails it mid-reasoning). This is the best answer to "your null is uninformative".
- **What would change my mind?** (i) F1 shows a J-lens advantage over the logit lens is not ruled out in many cells; (ii) W finds a plan in the reward-hack reasoning; (iii) EA27 passes the within-prompt control on 27B; (iv) a content-token readout sees the correct answer mid-reasoning in nudged runs. Say it on a slide.
- **Relation to the J-lens paper:** the paper's claims on Claude models (eval awareness shown in the lens) are not contradicted: we test a different model, a different lens. Our 14B ablation shows the causal effect can be reproduced; the specificity to the J-lens is what fails.
- **Is the method warning (cross-model lens comparisons measure the model) new?** Do not claim novelty. In docs/literature/ the closest items are the eval-awareness probe-control papers (format/surface confounds, Devbunova 2026; Nayan 2026) and model_quirks.md (adapter-vs-base setups); no grep hit for a cross-model-comparison confound in lens work. The general idea (a score separating two models can reflect model differences, not behaviour) is standard in the model-diffing and probing-control literature, so present it as "a trap we fell into, with a clean control", not a finding. The new bit is the number: the lens amplifies the shift more than the logit lens (0.94 vs 0.59 on the same text).
- **Statistics:** about 30 experiments, most exploratory, choices made after looking; four definitions of "known". Name the pre-registered ones.
- **Unreviewed analysis code** (agent-written): a hostile question about reproducibility; code is in the repo; only the planted-claim script was reviewed.

## 4. Experiments, ranked by value per hour (until ~08:00 GMT tomorrow, then after)

Resources: CPU here; octavia and derpy are busy; about $2 API.

| Rank | Experiment | Firms up | Cost | Fits before talk? | A result that changes the story |
| --- | --- | --- | --- | --- | --- |
| 1 | **F1 equivalence tests** on stored readouts (±0.10 AUROC, paired J − logit) | Turns "no advantage" into a bounded statement; every lens-null slide | CPU, 2-3 h | Yes (needs octavia files read-only; check the 27B case, `runs.npz`, can be read while GPU busy) | If many cells are inconclusive, change "no better" to "no evidence of better, power low". If a cell shows J-lens advantage ≥ 0.10, add it as a lead |
| 2 | **F2 difficulty control** on planted claim (0.71) | Whether the one positive is about the run or the question | CPU, ~1 h (reads known_items on octavia) | Yes | If it fails (explained by difficulty): drop the positive from the talk; if it survives within-question: promote to "lead worth a repeat" |
| 3 | **Slide-wording audit** (no compute): fix the five risky sentences in section 1 | Accuracy | 0.5 h | Yes | n/a |
| 4 | **Gate of the Qwen3-14B test-awareness run** (already running) | "Awareness is not an excuse", plus data for the within-prompt probe test | Running | Partly | If "testing me" runs take the bug less/more by a clear margin, flip or confirm the motivation caveat |
| 5 | **Cheap statistic on stored hack data**: bootstrap CI and equivalence bound for the one-model-fixed hack result (F1 case 4) | The "nothing separates within model" claim | CPU, in F1 | Yes | If upper bound is > 0.6, say "not excluded" |
| 6 | **Judge run on silent/open** with a "says it relies on the hint" criterion | "The reasoning says so" | ~$2 API plus 50 user labels | Run but unvalidated: label as such | If many mentions are not reliance, reword the headline to "names" only |
| 7 | **27B eval-awareness (EA27)** (running) | Slide 7 | Running | Maybe | A pass on the within-prompt control reverses the eval-awareness conclusion |
| 8 | **Wording sensitivity on other families** (running) | Slide 2 generality | Running | Maybe | If other families are insensitive to wording, say "Qwen-specific" |

After the talk:

| Rank | Experiment | Cost | Value |
| --- | --- | --- | --- |
| 1 | **W** reward-hack waterproofing (pre-registered) | 3-4 GPU-h | Decides "absent vs hidden in the code" for the headline hack result |
| 2 | **Nguyen-probe control** (design ready, 9 decisions open) | ~2 GPU-h | Turns the literature claim about probes into our own |
| 3 | **Content-token readout** mid-reasoning | ~2 GPU-h | Resolves the instrument limit behind every lens null |
| 4 | **Judge validation** (50 user labels) | $2-5 + 2 h | Every behavioural label |
| 5 | **Gemma-3-12B silent swap** under a lens (HF token) | ~1.5 GPU-h | The one place the lens might beat the chain of thought |
| 6 | Steering rerun at the home cap | ~2 GPU-h | Closes a deviation |

How to read: before the talk only ranks 1-3 are certain. The rest depend on running jobs and should be treated as footnotes on the slides.
