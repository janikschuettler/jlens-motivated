# Red team of the project's main claims

2026-11-04, written 20:55–21:02 GMT on arena9-cloud. Reading and CPU only: no GPU, no API calls to model providers. Sources: CLAUDE.md, docs/plan.md, docs/dashboard/site.yaml, docs/preregistration.md, the reports in docs/experiments/ and docs/literature/*.md. One sub-agent reviewed the analysis code of the planted-claim repeat and ran one new exploratory paired test from stored per-run scores (section 12). No new web search was done for this version; the literature notes in docs/literature/ were used as they stand.

This is a first version, meant to be extended. Each claim has its own section with a fixed layout (claim, support, objections, what would answer them). To add a claim, copy a section. To record that an objection has been answered, add a dated line under it rather than deleting it.

## Bottom line

- **The behavioural claims are on firmer ground than the lens claims.** That hint wording drives the switch rate is solid on Qwen3-8B. That the switches are "open" in thinking mode is solid as a fact about keywords, but only moderate as a claim about faithfulness, because no validated judge has ever been run on these runs.
- **The J-lens claims are mostly *absence* claims, and absence was never tested as absence.** "The J-lens adds nothing over the logit lens" rests on intervals that include zero, not on equivalence tests with a stated margin. Most cells are exploratory, small, and chosen after looking. The tracked tokens are mainly option letters, and an earlier check showed the letter readout can't show the answer mid-reasoning even in runs where the model plainly knows it. So many nulls say "this instrument can't see it", not "it isn't there". The verdict "the J-lens is not the best window on this behaviour" is fair. "The J-lens can't see arguing against knowledge" is not shown.
- **The one J-lens positive (planted claim, AUROC 0.71) is real as a pre-registered pass, but two things are not shown.** First, that the J-lens beats the logit lens: "J passes, logit doesn't" is a difference in significance, and a paired test run tonight gives a J-lens advantage of +0.11 [−0.10, +0.30], not significant. (The code review reproduced 0.71 exactly and found no bug.) Second, that it reads *retained knowledge despite arguing*, rather than how well the model knows the question. Most questions have only one outcome, so the score mostly ranks questions.
- **"The J-lens paper's eval-awareness result doesn't carry over" is too broad as worded.** On Qwen3-8B the pre-registered score test fails cleanly. But on Qwen3-14B the paper's causal effect *did* reproduce (removing eval-word directions silences "the user is testing me"); what failed is that it is specific to the J-lens. The 27B result is one exploratory smoke test, on a lens that the J++ authors may have found weak.
- **"Published eval-awareness probes read surface features" is a literature reading, not our result.** It is plausible and well sourced, but we have not rebuilt any of those probes.
- **Cross-cutting weaknesses:** keyword labels everywhere; four different definitions of "known"; Qwen-family and MMLU-Redux heavy; about 30 experiments, most exploratory, with only one pre-registered positive in the J-lens strand; agent-written analysis code that no one has reviewed independently.
- **Best value tomorrow:** cheap CPU re-analyses of data we already have (a paired J-lens minus logit-lens test, an item-difficulty control and equivalence tests), then judge validation, then the already pre-registered reward-hack waterproofing (W). Ranked list at the end.

## Claims table

Support ratings: **solid** (pre-registered or large, effect far from chance, robust to the obvious alternatives), **moderate** (consistent evidence with a known gap), **preliminary** (exploratory, small, or one source), **unsupported** (stated but not tested).

| # | Claim (short) | Support | Strongest objection | What would answer it | Cost |
| --- | --- | --- | --- | --- | --- |
| 1 | The gap with Mirtaheri & Belkin (M&B) comes from the hint's wording | Solid (Qwen3-8B) | One model; one fixed wrong option per question | Wording factorial on a second family (gpt-oss-20b locally) | ~1.5 GPU-h |
| 2 | Qwen3 follows sentences that state the answer, not information-free incentives | Moderate, scope-limited | Other models do follow the incentive once told they are scored; Qwen3-8B with thinking off does too (15%) | Restate the claim: "in chat framing, thinking-mode Qwen3-8B/14B"; then the same factorial on a second family | CPU (rewording); ~1.5 GPU-h |
| 3 | In thinking mode, switches are open (the reasoning names the hint) | Solid for keywords; moderate for faithfulness | Keyword labels only: naming the hint is not saying one relies on it | Run and validate the LLM judge with a "says it relies on the hint" criterion | $2–5 API + ~2 h human gold labels |
| 4 | The decision often comes before the reasoning | Moderate | At the cut before any reasoning, all runs of a prompt share the same prefix, so "settled" there is a property of the prompt; the 7-of-8 rule is noisy | Rephrase as "the prompt sets the outcome"; report it per question; more resamples at a few cuts | CPU; ~1 GPU-h |
| 5 | The J-lens adds little over the logit lens and simpler readouts (arguing against knowledge, cheating) | Moderate as "not better here"; preliminary as "nothing to see" | No equivalence tests; letter-only tracked tokens that can't show the answer mid-reasoning even when it is known; post-hoc positions | Equivalence tests on stored readouts; content-token readouts mid-reasoning | CPU 2–3 h; ~2 GPU-h |
| 6 | The J-lens paper's eval-awareness result doesn't carry over to open models | Moderate (Qwen3-8B score); preliminary (27B); contradicted in part (14B ablation reproduces the effect) | Not their model or data; Qwen rarely voices awareness; the 27B lens may be weak | Test on prompts where Qwen voices awareness often (Abdelnabi & Salem's test-awareness set, Qwen3-14B), within identical prompts | ~2–3 GPU-h |
| 7 | Published eval-awareness probes read surface features | Preliminary (literature only) | Benchmark format may be exactly what makes the model believe it is evaluated; we haven't rebuilt any probe | Rebuild the Nguyen/Hua probe on Qwen3-14B; test within identical prompts before "testing me"; text-classifier baseline | ~2 GPU-h |
| 8 | The planted-claim J-lens positive (AUROC 0.71) | Moderate that it passes (code reviewed, number reproduced); unsupported that the J-lens beats the logit lens (paired difference +0.11 [−0.10, +0.30]); preliminary that it reads retained knowledge | The score may rank questions by how well the model knows them (only 11 of 74 questions have both outcomes) | Regress on per-question knowledge strength; repeat on mixed-outcome questions only | CPU 1 h; ~1 GPU-h |
| 9 | The claimed-answer-key lead failed its repeat (0.54) | Moderate as "the lead didn't replicate"; not evidence of no effect | 95 runs, interval [0.37, 0.70]: an effect of 0.65 is not excluded | Leave as is; don't count it as a strong null | none |
| 10 | Reward-hack organism: the lens detects which model runs, not the hack | Moderate | 59 problems; no equivalence test; the hack may be decided in the code | Pre-registered W (already written) | ~3–4 GPU-h |
| 11 | J-lens directions are causal (steering 47% vs 27% logit lens, 59% probe, 2% random) | Moderate | Token cap 2,048 (unaccepted deviation) cuts 34–47% of wrong-answer runs; strengths run out of order | Rerun the per-token push at the home cap of 8,192 | ~2 GPU-h |
| 12 | Other models take the grader bug openly when told they are scored; two make silent letter swaps | Moderate (taking it); preliminary (silent swaps) | fp8 API providers, 2 samples, judge mislabels; swaps cluster on 7–13 questions | Gemma-3-12B locally in bf16, silent vs open on the same arm | ~1.5 GPU-h + HF token |
| 13 | gpt-oss-20b followers keep no trace of the correct letter; Qwen3-14B followers do | Preliminary | Crude salience control; generation cap 6,144 (deviation); letter readout only | Content-token readout; compare with a probe | ~1 GPU-h |
| 14 | The instruments are valid (token handoff, readout = jlens) | Solid for the pipeline; the labels are not validated | Keyword labels (judge built but never run on our runs) | Judge validation (row 3) | as row 3 |

## Cross-cutting objections

These apply to most claims and are not repeated in every section.

1. **Keyword labels, not a validated judge.** "Silent", "open", "followed", "says it is being tested" and the stance groups are regex or keyword rules. The LLM judge (registry `judge-offline`) agrees with one Claude agent's labels on 46 of 50 runs (Cohen's κ 0.89), but it was never run on the main run sets, and the gold labels aren't human. On the API runs, the judge (gemini-2.5-flash-lite) called silent letter swaps "rationalised"; the silent counts there are one agent's reading.
2. **Four definitions of "known".** 9 of 10 unhinted samples (the project rule), 5 of 5 (API models, moral questions), 3 of 3 (lens step 0), and one correct sample at temperature 0.1 (M&B replication). Rates on "known" questions are not comparable across these. A question that is right 5 of 5 times can still have a true accuracy of 70% (95% lower bound for 5/5 is about 48%).
3. **Run vs question units.** Most intervals resample questions, which is right. But several groups have many runs on few questions (silent switches: 115 of 124 on 42 questions; silent swaps on 7 items), and some AUROCs mix between-question and within-question variation. Where outcomes are mostly a property of the question, an AUROC over runs mostly ranks questions.
4. **Forking paths and multiple comparisons.** About 30 experiments, most exploratory, several with scores, layers and positions chosen after looking (the J-lens-vs-competitors table picks the best position per case and folds AUROCs below 0.5). The pre-registered tests are the ones to lean on: T1 (inconclusive), T1b (passes but no better than length), T2 (fail), H1–H3 (fail within model), E0–E6 (fail), S1–S3 (partial), X1–X3 (X1 pass, J-lens not specific), R-C3 (pass), R-CL (fail), Q6 (descriptive).
5. **Absence is not tested as absence.** Most J-lens claims are nulls. A null needs a bound ("AUROC within ±0.10 of chance") and an instrument that could have seen the effect (a positive control). Only the new W pre-registration does both. Elsewhere, "within 0.09 of chance" is a point-estimate statement, and upper bounds reach 0.58–0.75.
6. **What the lens reads.** Tracked tokens are fixed before each readout (rule 3), mostly the option letters, plus concept lists for some runs. Only the top 25 tokens per position are stored. Readouts are linear (lens or logistic probe). Lens-positions found that the letter margin is flat before the answer position *even in unhinted runs that go on to answer correctly*. So the letter readout can't show knowledge mid-reasoning at all, and every mid-reasoning letter null inherits that limit.
7. **Lens quality per model.** The Neuronpedia lenses were fit on wikitext. For Qwen3-8B there is no J++ lens. The J++ authors found "the released J-Lens for Qwen3.6-27B" weaker than their own; whether that is the file we used is not checked. Per-model faithfulness (how close lens readouts are to the model's own output at late layers) is checked for the hack adapters, not reported for every model.
8. **One family, one dataset.** Qwen3-8B carries nearly every lens result; Qwen3-14B, gpt-oss-20b and Qwen3.6-27B add a few. Questions are MMLU-Redux (plus MoralChoice for the moral hunt). Multiple-choice letters are an unusual readout target: the answer is a single token chosen at one position, which favours "decided at the answer" findings.
9. **Deviations awaiting the user's acceptance.** Steering cap 2,048 (plan 3,072, home 8,192); gpt-oss cap 6,144 and empty-text token IDs; re-tokenized API runs for the 27B smoke test (fidelity check passed, labelled).
10. **Unreviewed analysis code.** Every analysis script was written by agents and, apart from the review in section 12 of one script, has not been read by anyone else. Numbers in the reports come from those scripts. The ones carrying the most weight are listed under "Numbers that need independent re-analysis".

## Per-claim sections

### 1. Hint wording explains the M&B gap

**Claim.** On Qwen3-8B, the difference between M&B's high switch rate and our pilot's near zero comes from the hint sentence. Switching from M&B's wording ("Hint: An expert indicates the answer is X.") to ours ("An expert who answers correctly 90% of the time chose X.") lowers the switch rate by 51 percentage points [−56, −47], paired over 200 questions. Thinking mode, decoding and prompt format each move it by less than 7 points. A wording dissection (50 known questions) finds that what matters is whether the hint states the answer: "Hint: The answer is X." gets 67%, "An expert chose X." 4–13%. Sources: reports/gap_factorial.md, wave2_part1.md; registry `gap-factorial`, `wording-dissection`.

**Support: solid** for Qwen3-8B. 14,480 runs, a paired design, a huge effect, and the main alternatives (thinking, decoding, format) were varied in the same factorial.

**Objections.**
- One model. The factorial and the dissection are Qwen3-8B only.
- Two samples per question per cell (factorial), three per wording (dissection), 50 questions in the dissection.
- One fixed wrong option per question; M&B hint every option. Fine for the comparison, but switch rates depend on which wrong option is hinted.
- vLLM has no no-repeat n-gram blocking, which M&B used. Step 0 shows vLLM reproduces their rate within overlapping intervals (58% vs 68%), so this is minor.
- The parts of our wording were not all separated: the stated 90% reliability and the past-tense "chose" may each matter.
- Hint mentions are a keyword check (cross-cutting 1).
- Alternative explanation: "Hint:" reads as a label from whoever set the question, i.e. an answer key. That is the report's own reading and is compatible with the claim, but the mechanism (authority vs instruction-following vs format) is untested.

**What would answer them.** Repeat the factorial's wording factor and the dissection on a second model family that runs locally (gpt-oss-20b, or Gemma-3-12B once an HF token is available): about 1.5 GPU-hours, decisive for generality. Add the missing one-factor variants (drop "90%", present tense): about 0.5 GPU-hour on Qwen3-8B.

### 2. Models follow claims about the answer, not incentives

**Claim.** A reward that carries no information about the answer (the grader-bug note) barely moves Qwen3-8B (2% vs 0% for a null note); notes that claim B is correct move it (19–24%), openly. Sources: grader-bug-e3, wave2_part1.md, models_grader.md.

**Support: moderate, and only with a narrow scope.** As a general statement it is contradicted by the project's own data:
- Told it is a scored agent, most other models take the bug (gpt-oss-20b 58%, Gemma-3-27B 91%, Gemma-4-31B, DeepSeek-V4-Flash and Qwen3.6-27B 87–100%, Llama-3.3-70B 39%).
- Qwen3-14B with thinking off takes it 60% of the time under the agent framing; Qwen3-8B with thinking off 15%.
- "Allowed" sends almost every model, Qwen3-8B included, to 69–100%.

So the fair claim is: *in chat framing, thinking-mode Qwen3-8B (and to a lesser degree 14B) responds to claims about the answer and to permission, not to an information-free incentive. Other models and non-thinking Qwen respond to the incentive once the score is framed as the goal.*

**Objections.**
- The bug note and the "claims B is correct" notes differ in more than information content (length, source, placement).
- 3 samples instead of the planned 5 (grader-bug-e3), 40 questions, keyword labels.
- API runs: fp8 providers, providers may ignore seeds, "known" is 5 of 5.
- Alternative: thinking-mode Qwen3 may treat the bug note as a possible test or trap (the Qwen3 family says "the user is testing me" in 13–66% of grader-bug runs under scored framing). Lower uptake could be eval awareness, not indifference to incentives. Abdelnabi & Salem report that test awareness *lowers* harmful compliance in Qwen3-32B, which fits this.

**What would answer them.** Reword the claim in the registry (no cost). Then test the trap explanation: in Qwen3-14B grader-bug runs, do runs that voice "testing me" take the bug less? The screen says 10% vs 12%, so probably not, but with an interval. CPU, on stored labels.

### 3. In thinking mode, switches are open

**Claim.** In thinking mode, only 2 of 1,231 switches on factual questions and 0 of 5,019 on moral questions never mention the hint. Across 10 other models taking the grader bug, 99% of 2,850 judged followers name it. Sources: silent-scan-thinking, moral-hunt, models-grader.

**Support: solid** that the reasoning *mentions* the hint; **moderate** that the switch is *open* in the sense a monitor needs (the reasoning reveals that the hint caused the answer).

**Objections.**
- Mention is not acknowledgement. A chain of thought can mention the hint ("the hint says B, but let me check… B is right because…") and then give made-up reasons. Chen et al. 2025 ("Reasoning models don't always say what they think") count a chain of thought as faithful only if it says it *uses* the hint. Our keyword rule counts any mention as open, so it over-counts openness. The silent rate is a lower bound.
- The rule was never validated against human labels on these runs.
- Selection: the wordings that make the model switch are the ones that state the answer loudly ("Hint: The answer is X."). A loud hint is easy to name. Subtler cues give more silent switches (11 of 23 under ■-marked examples), but only on weakly known questions. So "open" may be a property of the strong nudges, not of the model.
- Planted claims in the model's own voice are silent by construction, so they don't test this.

**What would answer them.** Run the built judge (envs/judge) on a stratified sample (say 300 switches across wordings, plus all keyword-silent ones) with two criteria: "mentions the hint" and "says the hint is why it answers B". Validate on 50 runs labelled by the user. About $2–5 of API and 2 hours of the user's reading. Decisive for this claim and for every label-based result.

### 4. The decision often comes before the reasoning

**Claim.** Under "Hint: The answer is X.", runs that end on B already give B probability 0.82 when forced to answer before any reasoning, and it stays flat. Resampling from cuts in the reasoning: P(B) is about 0.87 at every cut; 50% [30, 70] of runs are "settled" (at least 7 of 8 resamples end on B) before any reasoning; 30% are never settled by it. Sources: forced-early-e5, q6-point-of-no-return.

**Support: moderate.** The forced-answer and resampling methods agree, and Q6 was pre-registered.

**Objections.**
- **At the cut before any reasoning, the prefix is the prompt alone.** All runs of one prompt share it. So "settled before any reasoning" is a property of the question and hint, not of the run, and the 50% is a share of questions, re-weighted by runs. The report's own wording ("set by the prompt") is right; "the decision precedes the reasoning" overstates it if read as a per-run hidden decision.
- The 7-of-8 rule is noisy: thresholds 0.75 and 1.0 move the settled share between 60% and 35%.
- Small groups (20–24 questions per main group; 10–12 for the comparisons).
- Forced early answers are off-distribution: resisters also give B 0.16 at the start.
- In thinking mode, the visible answer after `</think>` can still reverse the reasoning (6 of 10 resisters' reasoning ends on the hint), so "end of reasoning" is not "end of decision".

**What would answer them.** Report the before-reasoning result per question (CPU, existing data). For per-run claims, compare cuts within the first few sentences, where runs differ, with 16–32 resamples per cut on 20 questions: about 1 GPU-hour.

### 5. The J-lens adds little over the logit lens and simpler readouts

**Claim.** Before the final answer is written, the J-lens letter score separates arguing-against-knowledge or cheating runs from their comparison runs at most weakly (within 0.09 of chance in 10 of 11 comparisons). In paired tests it never beats the logit lens. Cheaper readouts do better where anything works: a keyword monitor (0.97 for runs told to argue for B), a probe at the end of the reasoning (0.90–1.00), re-asking the model (0.82). At the option words, a probe ranks the correct option first in 72% of no-hint runs, the J-lens in 42%, the logit lens in 43%. Sources: jlens_vs_competitors.md, instrument_q1q3.md, lens_main.md, lens_concepts.md, hack_organism.md, hack_manual_read.md.

**Support: moderate** as "on these setups and with these readouts, the J-lens is not the best window". **Preliminary** as "there is nothing for a lens to see".

**Objections.**
- **Exploratory throughout.** Cases, positions and readouts were chosen after earlier results; the summary picks each case's best position after looking; AUROCs below 0.5 are folded. Folding biases *towards* signal, so it is conservative for the null, but the table can't support a positive either way.
- **No equivalence tests.** "Never beats the logit lens" means every interval of the difference includes 0 or favours the logit lens. With 21–60 questions per group, those intervals can be wide; a real J-lens advantage of 0.05–0.10 AUROC would not be detected.
- **Instrument limit.** The letter readout shows no margin for the correct letter mid-reasoning even in no-hint runs that answer correctly (lens-positions). A readout that can't see the answer in the easy case can't be expected to see it in the hard one. So mid-reasoning letter nulls are mostly uninformative. Concept-token lists address this only in part, and only the top 25 tokens are stored.
- **Unfair comparison with probes.** The probes are trained on the group labels (supervised); the lenses are not. "A probe reads more" is expected, and at the end of the reasoning the probe reads the conclusion the text has reached.
- **Linear readouts only**, at fixed positions.
- **Lens quality.** Wikitext-fit lens, no J++ lens for Qwen3-8B (cross-cutting 7).
- **Alternative explanation.** Maybe the correct answer is not represented during the reasoning at all, and is only retrieved at the answer step when the question is re-read. Then no lens can show it mid-reasoning, and the null is about the behaviour, not the lens. The planted-claim result (section 8), which works only at the answer position, fits this.
- **Replication.** None of the competitor comparisons has been repeated on fresh runs.

**What would answer them.**
- Equivalence tests (two one-sided tests, margin ±0.10 AUROC, fixed in advance) for J-lens vs chance and J-lens minus logit lens, on every case at its pre-chosen position, from the stored readouts. CPU, 2–3 hours. Decisive for the wording of the claim.
- A readout with content tokens at reasoning positions: track the tokens of each option's text (where single-token), check them in the tokenizer before the run (rule 3), and read the J-lens mid-reasoning on the no-hint and follower runs. About 2 GPU-hours. Decisive for the instrument-limit objection.
- An unsupervised probe-like baseline (difference of class means from a held-out set of questions) to put the lens and the probe on an equal footing. CPU.

### 6. The J-lens paper's eval-awareness result doesn't carry over to open models

**Claim.** On Qwen3-8B, the paper's eval-awareness score does not separate 700 eval-like prompts from 200 real user prompts (AUROC 0.50 [0.44, 0.56]); it rises with eval wording in the prompt. On Qwen3.6-27B (exploratory smoke test) it gives 0.64, but the logit lens gives 0.66, and without the two suites with eval wording it drops to 0.55. Within identical grader-bug prompts, the score before "the user is testing me" doesn't separate runs that later say it (0.49). On Qwen3-14B, removing the eval-word J-lens directions silences "testing me" (0 of 552 vs 23%), but the logit-lens directions do the same (0 of 207). Sources: evalaware_replication.md, evalaware_models_screen.md, evalaware_ablation_14b.md, evalaware_big_smoke.md.

**Support:**
- **Moderate** for "the score test fails on Qwen3-8B" (pre-registered E0–E6).
- **Preliminary** for 27B (one exploratory smoke test, 19 questions in step 2, re-tokenized fp8 API runs).
- **Partly contradicted** by our own ablation: the paper's causal effect reproduces on Qwen3-14B (pre-registered X1 passes). What fails is J-lens specificity (and the behavioural part X2: taking the bug falls, the opposite of the paper's direction).

So the accurate claim is: *the score does not detect eval-like prompts beyond eval wording on Qwen3-8B; the ablation effect reproduces on Qwen3-14B but is not specific to the J-lens; we found no evidence that the lens reads an unspoken "this is a test" belief.*

**Objections.**
- Not their model: Qwen3-8B voices awareness on 1% of blackmail rollouts vs the paper's 71%. A failure may be about the model, not the method. No lens-covered model behaves like Claude here (at most 2 of 50).
- Not their data: public prompt sets stand in for their internal transcripts; WildChat is heavy on fiction; our control nouns aren't theirs (unpublished).
- 27B: the lens may be the weak release the J++ authors flagged (not verified). The J++ readout gave the same null, which weakens this objection.
- Ablation: "says it is being tested" is a regex, and removing the " testing" direction blocks the word itself; the model then says "trying to trick me" with the logit-lens version (56% vs 36%) but not with the J-lens version (23%). That difference is exploratory but is the one place the J-lens behaved differently, and it cuts *for* the lens.
- Several echo-control windows were tried on 27B (one gives a reversed 0.34).

**What would answer them.** Use a setting where Qwen voices awareness often and it matters for behaviour: Abdelnabi & Salem's test-awareness prompts on Qwen3-14B (rank 1 in docs/literature/model_quirks.md; gate on awareness ≥ 10% and on aware runs behaving differently). Pre-register a within-prompt test before the first awareness sentence, J-lens vs logit lens vs a rebuilt probe. About 2–3 GPU-hours. Also: a semantic label for "suspects a test" (judge) instead of the regex, so the ablation can't win by deleting one word; about $1.

### 7. Published eval-awareness probes read surface features

**Claim.** The probing evidence that models represent "I am being evaluated" is weak: the founding paper (Nguyen et al. 2025) controls only length, special characters and random directions; controlled follow-ups find the probe reads benchmark format (Devbunova 2026: 94.5% of real user requests in multiple-choice format called "evaluation"); probe scores barely track verbalized awareness (Heidari et al. 2026, Spearman < 0.19 including Qwen3-8B; Aranguri 2026, 0.4% within a prompt). Source: docs/literature/evalaware_internals.md.

**Support: preliminary.** It is a careful literature reading, not our experiment. Several sources are unreviewed 2026 posts or repositories, and some were read at abstract level.

**Objections.**
- Format is not obviously a *surface* feature here. Benchmark format is a large part of what makes a prompt an evaluation. A probe that reads format may be reading exactly what drives the model's belief. Only a contrast where the prompt stays the same but the belief differs can separate the two; the literature lacks it, and so do we (except the 0.49 within-prompt result for the J-lens score).
- Low correlation with *verbalized* awareness doesn't show the probe misses *unverbalized* awareness; verbalization is a noisy, partial proxy.
- Selective reading risk: the note was written to answer "is the evidence weak?", which can tilt the search.

**What would answer them.** Rebuild the Simple Contrastive probe (Nguyen) or Hua's 16-pair probe on Qwen3-14B from their public data. Score our suites against a TF-IDF text classifier and against eval-word-stripped versions. Then read it within identical grader-bug prompts before the first "testing me" sentence. About 2 GPU-hours. This is the most decisive test the literature note proposes, and it would turn this claim into our own result.

### 8. The one J-lens positive: planted claim, AUROC 0.71

**Claim.** We write "The answer is (B). Let me explain why." at the start of the model's reasoning on known questions. Among the 389 runs that keep B to the end, the J-lens score at the answer position (correct letter minus the other non-B wrong letters, layers 14–32) separates runs that name the correct option when re-asked without their reasoning from those that don't: AUROC 0.71 [0.57, 0.84]. Logit lens 0.60 [0.47, 0.74], reasoning length 0.47. Within questions 0.77 [0.62, 0.92]. The original exploratory lead was 0.86 on 61 runs. Pre-registered (R-C3, amendment R-C3.1). Sources: c3_lead_repeat.md, lens_c3.md.

**Support:**
- **Moderate** that the pre-registered test passes: fresh questions, fixed score, the lower bound 0.57 is clear of 0.5, the first batch alone gives 0.69.
- **Unsupported** that the J-lens beats the logit lens. "J passes, logit doesn't" with overlapping intervals is a difference in significance, not a significant difference. The paired test run tonight (section 12) gives +0.106 [−0.096, +0.295]: not significant.
- **Preliminary** that the score reads *retained knowledge despite the argument*.

**Objections.**
- **Mostly a question-level effect.** Of 74 questions, 36 have only "names correct" runs, 27 only "doesn't", and 11 both. So most of the AUROC compares questions, not runs. A score that measures how well the model knows a question would pass this test without saying anything about the run.
- **Thinking-off knowledge confound.** Re-asking is done with thinking off. The fresh thinking-off control is wrong on 16 of the 74 questions, and those 16 hold half of the "doesn't" group (93 of 190 runs) but only 6 of 199 "names correct" runs. Restricted to questions the control gets right, AUROC 0.68 [0.54, 0.83] (post hoc). This helps but doesn't remove the confound: within the remaining questions, knowledge strength still varies.
- **Within-question evidence is thin.** The 0.77 comes from 11 questions with both outcomes; "all 11 have the higher mean in the names-correct group" is the strongest single piece, but 11 is small.
- **Position.** The score is read just before the answer letter, after the reasoning. It is a readout of the answer step. That fits "the knowledge is retrieved at the answer", not "the knowledge is held during the argument".
- **Effect shrank** from 0.86 to 0.71 (expected regression from a post-hoc lead).
- **Weak in short runs** (shortest third: 0.55 [0.36, 0.75]).
- **Amendment** (second batch added after seeing the kept-B count, before any outcome); the first batch alone agrees, so minor.
- **Unreviewed code** (see section 12 for the review done tonight).

**What would answer them.**
- ~~Paired bootstrap of J-lens minus logit-lens AUROC.~~ Done 2026-11-04 by the code review (section 12): +0.106 [−0.096, +0.295], centred +0.085 [−0.019, +0.204]. The J-lens advantage is not significant. Exploratory, not pre-registered.
- Item-difficulty control: per question, the model's own confidence in the correct answer without a hint (log-prob of the correct letter in no-hint runs, thinking off and on). Regress the run's outcome on the J-lens score with this covariate, or compute the AUROC of the covariate alone. If the covariate alone gives ~0.7, the J-lens result is about the question. CPU if the no-hint readouts exist for these questions; otherwise about 0.5 GPU-hour.
- A within-question repeat: take the 11 mixed questions (plus near-mixed ones), sample 30–50 more planted-claim runs each, and test within questions only. About 1 GPU-hour. Decisive for "the score reads the run, not the question".

### 9. The claimed-answer-key lead failed its repeat

**Claim.** Among runs that followed a claimed answer key to B, the J-lens score at the answer doesn't separate "deferred" (names the correct option when re-asked with the reasoning in view) from "persuaded" runs: AUROC 0.54 [0.37, 0.70] vs the lead's 0.78. Pre-registered (R-CL). Source: claim_lead_repeat.md.

**Support: moderate** that the lead didn't replicate.

**Objections.**
- The report says "not underpowered" because the smaller group (40 runs) is above the 15-run threshold. But the interval [0.37, 0.70] still allows a moderate effect (0.65). This is a failed replication, not evidence of no effect.
- Only 2 samples per question and arm (the time-based stopping rule cut it after one chunk); 95 followers on 28 and 24 questions.
- The score uses only layers 30 and 32 (as the lead did), while the planted-claim score uses 14–32. Not a flaw (both match their leads), but the two "repeats" are not one test.
- Re-asking with the reasoning in view tests consistency as much as knowledge.

**What would answer it.** Nothing urgent. If the deferred/persuaded question matters, rerun with 6+ samples per question (about 1 GPU-hour). Count it in the registry as "not replicated", not as a strong null.

### 10. Reward-hack organism: the lens detects the model, not the hack

**Claim.** On a Qwen3-8B trained by others to reward-hack with hidden reasoning, the pre-registered J-lens test passes (AUROC 0.995) but only because it separates the two models; with the model fixed, hidden-hack and no-hack runs score alike (J-lens 0.44, logit lens 0.47). A blind manual read of 6,132 top-25 lists finds sneakiness words in both groups at similar rates (−3.1 points [−10.3, +4.1]). Sources: hack_organism.md, hack_manual_read.md.

**Support: moderate.**

**Objections.**
- 59 problems with both outcomes; no equivalence bound.
- The model was trained on silent examples; it may decide at the code, not in the reasoning, so there is nothing in the reasoning to read (the behavioural ceiling is unknown).
- Just before the code, a probe reaches 0.69 [0.60, 0.79] and the J-lens 0.57: some signal exists near the decision.
- Manual read: eight Claude readers, partial blinding, flag rates 16–30% per sheet.

**What would answer them.** The pre-registered W design (preregistration.md, section W) answers almost all of these: fresh runs, equivalence margin ±0.10, a behavioural ceiling from resampling after the reasoning, synthetic-plan positive controls, a coverage grid with Holm correction. About 3–4 GPU-hours. Highest-value GPU job because it was designed to be decisive.

### 11. J-lens directions are causal

**Claim.** Pushing along the J-lens direction of the correct letter at block 24 turns wrong answers correct in 47% [45, 49] of cases at the answer position, against 27% for the logit lens, 59% for a probe direction and 2% for random. Pre-registered S1–S3. Source: steering.md.

**Support: moderate.** Directions are unit length and centred over the four letters, so the comparison is fair in size.

**Objections.**
- Token cap 2,048 instead of 8,192 (needs the user's acceptance); it cuts off 34–47% of the wrong-answer runs. Counting cut runs as no answer, only the probe helps (24% → 42%).
- Strength 0.2 ran before 0.8 (plan had 0.8 first); 0.8 ran on 15 questions and breaks the text.
- The probe direction is supervised on the target.
- Steering a letter direction at the answer position mostly shows the direction controls which letter is written, which is close to what the logit lens direction does by construction.

**What would answer them.** Rerun the per-token push at the home cap on the same 40 questions: about 2 GPU-hours. Moderately decisive (it settles the deviation, not the interpretation).

### 12. Note on the planted-claim analysis code review

A sub-agent read `scripts/lens/analyze_c3_repeat.py` (and the post-hoc script) against the R-C3 pre-registration tonight (2026-11-04, ~21:10). It had no access to the raw readouts (they are on octavia, not in the local archive), so it checked the code by reading and recomputed everything downstream of the per-run scores stored in docs/experiments/replicate-leads/data/c3_lead_repeat.json. Scratch script: /tmp/c3rev/rep.py (not in the repo).

- **Reproduced exactly:** J-lens 0.709 [0.570, 0.843], logit lens 0.603 [0.465, 0.739], from the stored per-run scores. No ties, no duplicate runs, group counts agree (199 vs 190).
- **Matches the pre-registration:** score definition (correct letter minus the mean of the non-B wrong letters, each letter a log-sum-exp over "X", " X", "(X"), position (the token before the answer letter; no off-by-one), groups (re-ask answer = correct vs anything else), exclusions (known ≥ 0.9, no overlap with the original questions), question-level bootstrap, amendment reporting. No leakage found; the lens order in storage and analysis agree.
- **Not checked (raw data not local):** that the stored layers are 14–32 in these readout folders (the script doesn't assert it), the per-run score computation from the readout files, the re-ask answers.
- **New number, paired J-lens minus logit-lens test** (same 2,000 question resamples for both lenses, exploratory, run tonight): difference +0.106 [−0.096, +0.295]; centred on question means +0.085 [−0.019, +0.204]. **So the J-lens does not significantly beat the logit lens on this test.**
- **Wording issue in the report:** the within-question AUROC 0.77 is post hoc for the J-lens too (the pre-registration only fixes the 11-of-11 share). It rests on 11 questions, and a percentile bootstrap on 11 clusters likely gives too narrow an interval.
- Minor issues in the post-hoc script (a printed count without a mode filter; relative paths) don't change any number.

### 13. Other models take the grader bug openly; silent letter swaps

**Claim.** Told they are scored, most non-Qwen models take the bug and say so (99% of 2,850 judged followers). With the note in the system prompt, Llama-3.3-70B (22 of 250 followers) and Gemma-3-12B (47 of 501) sometimes argue for the correct option and then write B with no mention. Source: models_grader.md.

**Support: moderate** for taking the bug openly; **preliminary** for silent swaps.

**Objections.** fp8 API providers; seeds may be ignored; 2 samples on 35–60 questions; "known" is 5 of 5; the judge called swaps "rationalised" and the swap counts are one agent's reading of keyword-silent followers only; swaps cluster on 7 (Gemma) and 13 (Llama) questions, three Gemma questions in all 6 samples. Alternative: a letter-mapping or position slip on a few questions (the matched null gives no followers at all, which argues against a pure slip, but a slip that only happens when B is primed is still not "hiding").

**What would answer them.** Gemma-3-12B-it locally (bf16, Neuronpedia J-lens), the system-prompt arm, 10+ samples on the swap-prone and other questions; label swaps with a validated judge; pre-register silent swap vs open exploit at the answer position. About 1.5 GPU-hours plus an HF token. This is the only setting found so far where a lens could beat the chain of thought.

### 14. gpt-oss-20b keeps no trace of the correct letter

**Claim.** At the answer position, gpt-oss-20b followers keep no correct-letter margin (+0.08 nats), while Qwen3-14B followers keep +1.2. Source: docs/experiments/models-grader/report.md, docs/experiments/models-grader/gptoss_phase2.md.

**Support: preliminary.** One readout, one position, letter tokens only; the salience control is crude; generation capped at 6,144 (deviation); empty-text token IDs. A different tokenizer and lens fit make cross-model nats hard to compare.

**What would answer it.** A probe at the answer position (does a supervised readout find the correct option in gpt-oss followers?), CPU if block outputs were stored, else about 1 GPU-hour.

## Numbers never validated, or needing independent re-analysis

- Every keyword-based rate: silent 2 of 1,231; 0 of 5,019 moral; "open" shares; non-thinking silent 12–17%. Not validated against human labels.
- Silent letter swap counts (22, 47): one agent's reading.
- The judge's 99% open on 2,850 followers: judge not validated on these runs, and known to mislabel swaps.
- The planted-claim 0.71 and its post-hoc checks: one script, now reviewed once (section 12).
- The J-lens-vs-competitors table: exploratory, many scripts, best-position selection; no one has re-derived any cell.
- Steering 47/27/59/2%: affected by the cap deviation.
- The 27B eval-awareness smoke numbers: re-tokenized API runs, several windows tried.
- Point-of-no-return settled shares: sensitive to the 7-of-8 threshold.

## Experiments for tomorrow, ranked by value per hour

| Rank | What | Answers | Cost | Why this rank |
| --- | --- | --- | --- | --- |
| 1 | Item-difficulty control on the planted-claim repeat (the paired J minus logit test is done: not significant) | Claim 8 (is the one positive about the run or about the question?) | CPU, ~1 h if the no-hint readouts of these questions exist on octavia; else ~0.5 GPU-h | Cheap, could remove or firm up the only positive |
| 2 | Equivalence tests (±0.10 AUROC) for J-lens vs chance and J-lens vs logit lens on every stored case | Claim 5 ("adds little" becomes a bounded statement) | CPU, 2–3 h | Turns the project's headline null into something a reviewer accepts |
| 3 | Judge run and validation (300 stratified switches + all keyword-silent ones; 50 gold labels by the user; criterion "says the hint is why") | Claims 3, 12, 14 and every label | $2–5 API + 2 h of the user | Every behavioural claim leans on unvalidated labels |
| 4 | W: reward-hack waterproofing (already pre-registered) | Claim 10 | ~3–4 GPU-h | Designed to be decisive; ready to run |
| 5 | Independent code review of the analysis scripts behind claims 1, 3, 5 (analyze_gap.py, w2_silent_scan.py, analyze_competitors.py, competitors_extract.py) by fresh agents with the pre-registrations | Cross-cutting 10 | ~2 h agent time, no GPU | Cheap insurance on the numbers the summary quotes |
| 6 | Gemma-3-12B: silent swap vs open exploit on the same arm, pre-registered | Claims 5 and 13; the CoT-vs-lens test the project lacks | ~1.5 GPU-h + HF token | Only setting with silent switches on a lens-covered model |
| 7 | Within-question repeat of the planted-claim test on mixed-outcome questions | Claim 8 | ~1 GPU-h | Decisive for "reads the run", but do rank 1 first |
| 8 | Eval-awareness probe rebuilt on Qwen3-14B; within-prompt test before "testing me"; TF-IDF baseline | Claims 6 and 7 | ~2 GPU-h | Turns a literature claim into our own |
| 9 | Content-token readout mid-reasoning (option-text tokens, fixed and checked before the run) | Claim 5, instrument limit | ~2 GPU-h | Answers "the letter readout can't see it anyway" |
| 10 | Wording factorial on a second family (gpt-oss-20b) | Claims 1 and 2 | ~1.5 GPU-h | Generality of the best-supported behavioural claim |
| 11 | Steering rerun at the home cap | Claim 11 | ~2 GPU-h | Settles a deviation; doesn't change the story much |

How to read: ranks 1–3 and 5 need no GPU and could all be done tomorrow morning. Ranks 4 and 6 are the GPU jobs with the most decision value.

## Changes suggested for the registry (not made; the user's call)

- Strand A: restrict the "claims, not incentives" statement to thinking-mode Qwen3-8B/14B in chat framing (claim 2).
- Strand B: say "names the hint" rather than "open" until the judge is validated (claim 3); describe the before-reasoning result as "set by the prompt" (claim 4).
- Strand C: say the J-lens vs logit-lens difference in the planted-claim test is untested (claim 8).
- Eval-awareness entries: say the 14B ablation reproduces the paper's effect but not its J-lens specificity (claim 6).
