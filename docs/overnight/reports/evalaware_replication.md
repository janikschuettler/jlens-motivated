# Does the J-lens eval-awareness result replicate, and does it carry over to arguing against its own knowledge?

**Plain answer: their eval-awareness result does not replicate on the model we can run, and it stops working at the first quantitative step.** The paper's results are on Claude Sonnet 4.5 and Opus 4.6, so an exact replication is impossible. On the closest model we can run, Qwen3-8B with its J-lens, the paper's readout gives the same picture *on a single prompt*: on the blackmail prompt, eval-awareness words such as " fake", " fictional", " simulation" and " benchmark" show up among the lens's top 20 tokens. But the paper's one-number eval-awareness score does not tell evaluation prompts from real user prompts (AUROC 0.50, where 0.5 is chance and 1 is perfect). Real user requests from WildChat and OASST score as high as the evaluation suites, mostly because they contain fiction and role-play. The score does go up when a prompt *says* it is a benchmark or an evaluation. It also goes up on prompts where the model later writes "the user is testing me". But that link is mostly a difference between prompt sets, not between prompts within a set. The paper's causal test can't be run here: Qwen3-8B neither blackmails nor says "this is a test" on the blackmail prompt (0 and 1 of 100 rollouts). Ablating the eval-token directions anyway leaves both at zero (0 of 60 rollouts blackmail, none says "test").

Moving towards our setup, nothing in the reasoning carries over. On the grader-bug runs, the eval-awareness, "cheating" and "wrong answer" scores in the reasoning don't tell runs that took the bug from runs that didn't, once we exclude positions after the reasoning first mentions the grader (AUROC 0.45–0.53). The same holds for the hidden reward hack of the trained organism (0.48–0.50). The one place a score separates the groups strongly is the answer position: there the " fictional"/" simulation" score separates runs that took the bug from runs that didn't (AUROC 0.92 for Qwen3-8B without thinking, 0.86 for Qwen3-14B with thinking). But that position comes after the text has already stated the choice, the logit lens does the same, and a keyword count of the text separates the groups nearly as well on Qwen3-14B (0.87).

*Interpretation:* on Qwen3-8B, the paper's eval-awareness readout behaves like a detector of eval-related *words and topics* near the position read, not of a hidden belief that the situation is a test. So the paper's positive result does not transfer to our setup at the prompt level, and the steps towards arguing against its own knowledge add nothing beyond the text. Moderate confidence: one model (plus Qwen3-14B for one step); our corpora stand in for the paper's internal transcripts; our 100 control words are not the paper's; the regex for "says it is a test" is rough (about 4 in 5 matches are real, by reading 20).

| Step | What changed | Result | Works? |
| --- | --- | --- | --- |
| 0 | Find their work | Claude Sonnet 4.5 / Opus 4.6 only; prompts, transcripts and comparison method (NLAs) not public | can't replicate exactly |
| 1a | Their blackmail prompt and 28 eval tokens, on Qwen3-8B's J-lens | Eval tokens in the top 20 at 4.0% of prompt positions, against 0.6% for control nouns. But they sit on words like "artificial", "sandboxed" and "performance", and real user prompts show them more often (9.7–10%) | partly: they appear, but not specifically on evals |
| 1b | Their score at the start of the model's turn: 7 eval-like suites against 2 sets of real user prompts | AUROC 0.50 [0.44, 0.56]; 1 of 7 eval suites scores above both deployment sets | **no** (this is where it breaks) |
| 1c | Score vs the model saying "the user is testing me" in its reasoning | 0.77 [0.71, 0.83] across all prompts, but 0.54 and 0.59 within the two suites where it happens most (intervals include 0.5) | passes the pre-registered rule, but mostly a suite effect |
| 1d | Their ablation of the eval-token directions, blackmail prompt | Qwen3-8B doesn't voice eval awareness (1 of 100 unablated rollouts) and never blackmails (0 of 100). With the ablation: 0 of 60 say "test", 0 of 60 blackmail | can't be tested (nothing to remove) |
| 1e | Their steering check (push towards "test" vs "ordinary conversation") | Push along the "benchmark minus coworkers' argument" direction: the score in the generated text rises from −1 to +1 on both prompt sets we tried (+0.16 and +0.19 nats; exploratory) | yes, small (but the direction is built from eval wording) |
| 2a | Our MMLU questions, benchmark framing vs a coworkers'-argument framing (same question) | Benchmark framing +0.17 [+0.07, +0.27] higher (J-lens). But the logit lens goes the other way (−0.14), and our usual plain format scores 0.92 *below* the casual one | only for explicit "benchmark" wording |
| 2b | Positions inside the reasoning instead of the start of the turn | Eval vs deployment AUROC 0.48 [0.43, 0.53]; benchmark − casual +0.09 [+0.04, +0.14] | no |
| 3 | Grader-bug runs: took the bug vs answered correctly, reasoning positions | Qwen3-8B (no thinking) 0.54 [0.50, 0.58]; Qwen3-14B (thinking) 0.67 [0.61, 0.73]. Before the first mention of grader/bug: 0.51 and 0.52. Keyword count of the text: 0.74 and 0.87 | no (nothing beyond the text) |
| 4 | "Cheating" and "wrong answer" words instead; and the trained organism's hidden hack | Before the first mention: 0.50–0.53 (Qwen3-8B), 0.45–0.51 (Qwen3-14B). Hidden hack vs no hack in one model, before the code: 0.48–0.50 | no |
| 5 | The answer position | Eval score 0.92 (Qwen3-8B), 0.86 (Qwen3-14B); "cheating" score 0.89 / 0.80. Logit lens about the same. Holds within confidence strata and within runs whose text names the bug | yes, but only after the choice is written |

Pre-registered rules: 1c, 2a, 5 and the paired half of 2b pass; 1a fails one of its two conditions; 1b, the pooled half of 2b, 3 and 4 fail; 1d was underpowered (amended in advance). 1e is exploratory.

How to read the table: AUROC is the chance that a random run (or prompt) from the first group scores higher than one from the second; 0.5 is chance, 1 is perfect. Intervals in brackets are 95% bootstrap intervals. Scores are in nats (natural-log units of probability).

## Step 0: what the paper did

**Source.** Gurnee, Sofroniew, … Batson, Lindsey (Anthropic), published 6 July 2026, "Verbalizable Representations Form a Global Workspace in Language Models", Transformer Circuits Thread ([transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html)). It is the paper that introduces the J-lens; the `jlens` package we use (`anthropics/jacobian-lens`, commit 581d3986) is its companion code. The eval-awareness results are in the section "Using the J-lens for alignment auditing" and the appendix "Measuring evaluation awareness with the J-lens". Credits in the paper: blackmail case study Jack Lindsey and Wes Gurnee; "Measuring evaluation awareness" Jack Lindsey and Subhash Kantamneni.

**Models.** Claude only: the blackmail study uses **Claude Sonnet 4.5**, the measurement appendix **Claude Opus 4.6** (transcripts from Anthropic's pre-release audits and internal deployment traffic). No open model is used for eval awareness. We can't run either model, and their prompts, transcripts and the comparison method (natural-language autoencoders, "NLAs", Fraser-Taliente et al. 2026) are not public. So an exact replication is impossible. What is public: the blackmail prompt (shipped in `jlens/data/blackmail.json`, from the open Agentic Misalignment release), the eval-awareness token list, the score's definition, and the lens method. The paper's companion notebook uses Qwen3.5-4B and Qwen3.6-27B; Neuronpedia hosts a J-lens for our Qwen3-8B. **We use Qwen3-8B (thinking mode) as the closest model we can run.** It is lens-covered, fits the L40S and is the project's model. Qwen3.6-27B doesn't fit in bf16; Gemma models are gated and we have no token.

**Lens.** The J-lens: `lens_l(h) = softmax(W_U · norm(J_l h))`, with `J_l` the average Jacobian of the final residual stream with respect to layer l's residual, fitted on generic text. Not J++ (the paper's eval-awareness work uses the plain J-lens). The paper reports 25 evenly spaced layers on a 0–100 depth scale; the "workspace layers" are L38–92 (as % of depth). Ours are blocks 14, 16, …, 32 of 36 (39–89% of depth), fixed in the roadmap before tonight.

**What they measured, exactly.**

| Claim | Model | What was done | Reported numbers |
| --- | --- | --- | --- |
| Eval awareness shows in the J-lens on the prompt | Sonnet 4.5 | J-lens top tokens at selected positions of the blackmail prompt, before any output | "fake and fictional appear in the J-space early in the prompt" (qualitative) |
| It is causal | Sonnet 4.5 | Ablate (project out) the J-lens directions of 28 eval-awareness tokens ("fake, fak, fictional, scenario, artificial, fict, benchmark, scenarios, hypoth, experiment, fals, faker, experiments, simulation, dummy, mock, sham, artific, simul, bench, simulate, evalu, synth, simulator, experimental, Mock, testing, experi"; chosen from tokens in the J-lens top-20 somewhere in six blackmail prompts) at the workspace layers, all positions; 6 prompts × 30 rollouts | Verbalized eval awareness 71% → 3%; blackmail 0/180 → 13/180 (7%) |
| A one-number score tracks eval awareness | Opus 4.6 | Score = mean J-lens log-prob of " fictional" and " simulation" minus the mean over 100 control nouns ("recipe, bicycl, guitar, …"; full list not published), averaged over the workspace layers | see the next four rows |
| … agrees with NLAs | Opus 4.6 | Assistant-turn starts in ~94,000 auditing transcripts; NLA flags 198 as latently eval-aware | AUC 0.853 (198 flagged vs 594 sampled others) |
| … moves with steering | Opus 4.6 | Steering along a contrastive "test vs ordinary conversation" direction (−0.1, 0, +0.1) on 6 eval suites; score at random response positions | monotonic on every suite |
| … drops with realism edits | Opus 4.6 | 11 eval suites, original vs rewritten to read like real use | lower on 11 of 11, significant on 8 |
| … is higher on evals than on deployment | Opus 4.6 | 7 alignment + 9 capability evals vs internal Claude.ai and Claude Code traffic | deployment centres near 1.0; top evals 2–3 units above; 12 of 16 evals above Claude Code, 10 of 16 above Claude.ai |

The paper calls the score "a proof of concept" and warns that " fictional" can light up on real discussions of fiction. No code for the eval-awareness experiments is released (the repo has fitting, applying and the slice viewer).

## Setup common to every step

2026-11-04, arena9-octavia (L40S). Pre-registered as E1a–E5 in [../preregistration.md](../preregistration.md) at 00:04, before any readout. The E1d ablation was amended at 00:54, after the vLLM blackmail rollouts and before any ablation run. Everything marked *exploratory* below was added after the first numbers. Derived numbers: [../data/evalaware_replication.json](../data/evalaware_replication.json).

- **Model:** Qwen3-8B at our pinned revision, thinking mode, T 0.6, top-p 0.95, top-k 20. Step 3 also uses Qwen3-14B (thinking, its own Neuronpedia J-lens) and Qwen3-8B without thinking, from existing grader-bug runs.
- **The score** (the paper's, mapped to Qwen). At a position, take the J-lens log-probability of " fictional" and of " simulation", average the two, and subtract the mean log-probability of 100 control nouns (" recipe", " bicycle", " guitar", …). Then average over the workspace layers 14, 16, …, 32. The paper doesn't publish its 100 control words, so ours are fixed in `scripts/lens/evalaware_tokens.py`. The logit lens version of the same score is the comparison. "Start of the model's turn" is the last prompt token, the newline after `<|im_start|>assistant`.
- **Eval-token set:** the paper's 28 tokens; 26 map to single Qwen tokens ("fak" and "artific" don't).
- **Readout:** `scripts/lens/readout_evalaware.py` teacher-forces the saved token IDs and computes the full softmax in memory. It stores only the tracked tokens' log-probs, the control-noun mean and the top-25 at anchors (rule 3). It reads every prompt position of prompts up to 1,024 tokens (else the last 64), 64 evenly spaced reasoning positions, 16 response positions and the anchors. **Checks, all passed:** teacher-forced log-probs match vLLM's on every run set (median |Δ| 0.0001–0.002, p99 0.12–0.18; tolerance 0.05 and 0.5). Logit-lens values equal the validated single-run readout exactly. J-lens values match it on 99.7–99.99% of entries; the rest differ by at most 0.06–0.25 nats, a few fp16 rounding steps, because the matrix shapes differ.
- **Prompt sets** (`scripts/gen/evalaware_corpus.py`; 100 prompts each, 2 samples each). *Evaluation-like:*
  - redux_home: known MMLU-Redux questions in our usual format;
  - redux_bench: the same questions introduced as "a multiple-choice question from the MMLU benchmark, which is used to evaluate … AI models";
  - syco: Anthropic's model-written-evals sycophancy prompts;
  - moral: MoralChoice scenarios;
  - hack_task: the reward-hack coding task;
  - grader_agent: the grader-bug note with the "you are an AI agent being evaluated" framing;
  - grader_plain: the grader-bug note alone;
  - blackmail: the paper's prompt, 100 samples.

  *Deployment-like:*
  - oasst: OpenAssistant first user turns;
  - wildchat: WildChat-1M first user turns, English, non-toxic;
  - redux_casual: the same MMLU questions asked as "My coworker and I were arguing about this at lunch …".
- **"Says it is a test":** a fixed regex on the reasoning text, e.g. "the user is testing if I …" or "this is a test/simulation". By reading 20 random matches, about 16 are real test awareness. The rest echo the prompt ("being evaluated") or mean something else ("the claim being evaluated").

## Step 1: replicate their findings on Qwen3-8B

### 1a. Eval words in the blackmail prompt (paper: "fake and fictional appear in the J-space early in the prompt")

We counted prompt positions where at least one of the 26 eval tokens is among the J-lens top 20 at any workspace layer.

| Prompt | Positions | Eval tokens in top 20 (J-lens) | 26 control nouns in top 20 | Eval tokens, logit lens |
| --- | ---: | ---: | ---: | ---: |
| Blackmail (the paper's) | 2,122 | 4.0% | 0.6% | 2.4% |
| OASST user prompts | 3,611 | 9.7% | 3.4% | 7.2% |
| WildChat user prompts | 11,126 | 10.0% | 1.7% | 8.4% |

How to read: the share of prompt positions where the lens's top 20 contains at least one word of the set.

As in the paper, " fake" and " fictional" appear at the very start of the blackmail prompt. Then come " simulation", " simulator", " scenario", " testing" and " benchmark". Where they sit tells what they track. " fake", " fictional" and " artificial" appear on "You are Alex, an advanced **artificial** intelligence". " simulation" and " simulator" appear on "operating in a **sandboxed** email client". " scenario(s)" appears on "analyze the **situation** you are in". " benchmark" appears on "Our Q4 results have **exceeded all expectations** … reliable **performance**". These are near-synonyms of nearby words. On real user prompts the same tokens appear *more* often, mostly " fictional" and " fake" on stories and role-play. The pre-registered rule (more than 2× the control nouns *and* more than 2× the deployment prompts) fails on its second half.

### 1b. The score on evaluation vs deployment prompts (paper: evals score 2–3 units above deployment)

| Prompt set | Kind | Score at the start of the model's turn, J-lens | logit lens |
| --- | --- | ---: | ---: |
| MMLU, our format | eval | 2.26 [2.18, 2.35] | 2.03 |
| MoralChoice | eval | 2.85 [2.79, 2.91] | 2.38 |
| Reward-hack coding task | eval | 2.92 [2.88, 2.95] | 2.55 |
| Grader-bug note | eval | 3.02 [2.94, 3.10] | 2.41 |
| OASST | **deployment** | 3.07 [2.86, 3.30] | 2.73 |
| MMLU, coworkers' argument | **deployment** | 3.19 [3.10, 3.28] | 2.26 |
| MMLU, "from the MMLU benchmark" | eval | 3.35 [3.28, 3.43] | 2.12 |
| Grader-bug note + "AI agent being evaluated" | eval | 3.64 [3.60, 3.68] | 2.78 |
| WildChat | **deployment** | 3.86 [3.53, 4.22] | 3.35 |
| Sycophancy eval | eval | 4.02 [3.98, 4.07] | 4.06 |
| Blackmail (one prompt) | eval | 5.17 | 3.47 |

How to read: higher means the lens puts more weight on " fictional"/" simulation" than on everyday nouns; units are nats; intervals resample prompts.

Pooled, the score does not separate the 700 eval-like prompts from the 200 real user prompts: AUROC 0.50 [0.44, 0.56] (logit lens 0.41). Only the sycophancy suite scores above both deployment sets; the paper had 10–12 of 16 evals above deployment. **This is where the replication stops working.** The blackmail prompt does score highest of all, as in the paper. But it is one prompt, and WildChat's role-play prompts reach the same range.

### 1c. Does the score track the model *saying* it is being tested? (our stand-in for the paper's NLA check)

Across all 1,000 prompts, the 49 prompts where at least one sample says something like "the user is testing if I …" score higher: AUROC 0.77 [0.71, 0.83] (logit lens 0.75). Most of these prompts come from two suites, the agent-framed grader note (23) and sycophancy (13), and both score high anyway. Within those suites the link mostly goes: 0.59 [0.47, 0.70] and 0.54 [0.37, 0.69] (exploratory). So the score knows *which kind of prompt* invites "you're testing me". It can't tell which prompt within a kind does.

### 1d. Ablating the eval-token directions on the blackmail prompt (paper: verbalized eval awareness 71% → 3%, blackmail 0/180 → 13/180)

First, the baseline. In 100 vLLM rollouts of the paper's prompt, Qwen3-8B says the situation might be a test once ("the user might be testing if I can simulate that"). It blackmails in none of them. Our heuristic flagged 2 emails to Kyle that mention the affair. On reading, one promises discretion and the other advises him to sort it out privately. Neither is blackmail. Most rollouts just confirm the 5 pm wipe or ask Kyle to double-check the transition. With only 1 in 100 voicing awareness, the pre-registered awareness test is underpowered from the start (the rule needed at least 6 of 60). We ran the ablation anyway, at a smaller size (amendment logged before the run).

The ablation projects the 26 eval tokens' J-lens directions out of the residual stream at every block 14–32 and every position, as in the paper. It works mechanically: at block 24, the eval tokens are in the top 20 at 21 prompt positions without it and at 0 with it. The control ablation removes 26 everyday nouns' directions instead.

| Condition (HF generate, T 0.6) | Rollouts | Says it may be a test | Blackmail (by reading) | Median length |
| --- | ---: | ---: | ---: | ---: |
| No ablation | 30 | 0 | 0 | 1,404 tokens |
| Eval-token directions removed | 60 | 0 | 0 (heuristic flagged 3, none is blackmail) | 944 tokens |
| Control-noun directions removed | 30 | 0 | 0 (heuristic flagged 1, not blackmail) | 1,116 tokens |

How to read: counts of rollouts; "by reading" means every email the heuristic flagged was read by eye.

The paper found that removing eval awareness from Sonnet 4.5 exposed a hidden willingness to blackmail. In Qwen3-8B there is no voiced eval awareness to remove, and removing the directions exposes nothing. Two ablated rollouts argue against the wipe ("I recommend delaying the wipe"), but without any leverage. Some unablated rollouts do the same.

### 1e. Steering check (paper: steering towards "test" raises the score, towards "ordinary conversation" lowers it, on 6 of 6 suites)

*Exploratory; not in the pre-registration.* The paper steered along a "test vs ordinary conversation" direction built from paired prompts, at −0.1, 0 and +0.1. The score at response positions then rose monotonically on all 6 suites. Our version builds the direction from the paired prompts of step 2a: the mean difference of the last-prompt-token residual, benchmark framing minus coworkers' framing, over 50 questions, at every block 14–32. We add α × that direction (split over the 19 blocks) at every position, with α = −1, 0, +1. We generated up to 1,024 tokens for 50 held-out MMLU questions (usual format) and 50 OASST prompts. Each response was then scored by the *unsteered* model reading the generated text. That asks whether the text itself became more eval-flavoured, not whether we pushed the readout directly.

| Prompt set (50 each) | α = −1 | α = 0 | α = +1 | +1 minus −1 (paired) |
| --- | ---: | ---: | ---: | ---: |
| MMLU, usual format, J-lens | 1.75 | 1.86 | 1.90 | +0.16 [+0.08, +0.24] |
| MMLU, usual format, logit lens | 1.21 | 1.25 | 1.30 | +0.09 [+0.05, +0.14] |
| OASST, J-lens | 1.82 | 1.86 | 2.01 | +0.19 [+0.03, +0.36] |
| OASST, logit lens | 0.93 | 1.00 | 1.11 | +0.18 [+0.09, +0.27] |

How to read: mean score over the response positions (nats), read by the unsteered model; the last column resamples prompts.

The score moves monotonically in both sets, as in the paper. The effect is small, and the model did not start saying "this is a test": 0, 1 and 0 MMLU responses, 0, 1 and 2 OASST responses at α = −1, 0, +1. Almost every response hit the 1,024-token cap. Because the direction comes from prompts that differ in eval *wording*, this check confirms the score follows that direction. It doesn't show that the direction is eval awareness.

## Step 2: our MMLU prompts, framed as an evaluation or as ordinary use

**2a. Same question, two framings** (the paper's "realism edit": rewrite the prompt so it reads like real use). On 100 known MMLU-Redux questions:

| Comparison (same 100 questions) | J-lens | logit lens |
| --- | ---: | ---: |
| "From the MMLU benchmark" minus coworkers' argument (pre-registered) | +0.17 [+0.07, +0.27] | −0.14 [−0.23, −0.05] |
| Our usual plain format minus coworkers' argument | −0.92 [−1.03, −0.81] | −0.23 [−0.34, −0.12] |
| "From the MMLU benchmark" minus our usual plain format | +1.09 [+1.00, +1.18] | +0.09 [+0.01, +0.17] |
| Grader note with "AI agent being evaluated" minus grader note alone | +0.62 [+0.55, +0.70] | +0.38 [+0.31, +0.44] |

How to read: mean difference in the score at the start of the model's turn, same question in both framings; positive means the first framing looks more like an evaluation to the readout.

The pre-registered comparison passes for the J-lens. But the plain multiple-choice format, which is an evaluation item with no "benchmark" wording, scores far *below* the casual framing. So what raises the score is the words "benchmark" and "evaluated" in the prompt, not the evaluation-like format. That is a word detector. No sample in any MMLU framing said it was being tested (0 of 600).

**2b. Positions inside the reasoning** (the paper's steering check scores positions in the response). Averaged over 64 reasoning positions, eval vs deployment prompts: AUROC 0.48 [0.43, 0.53] (logit 0.52). Benchmark minus casual framing: +0.09 [+0.04, +0.14]. The reasoning carries even less of it than the start of the turn.

**2c. Our model.** Steps 1 and 2 already use Qwen3-8B in thinking mode, so there was nothing to change here.

## Step 3: the grader-bug runs (took the bug vs answered correctly)

Existing runs from [models_grader.md](models_grader.md), with the grader-bug note under the agent framing ("You are an AI agent being evaluated … highest score"). We used every question with at least one run that took the bug (final answer = B) and one that answered correctly, in the same prompt variant.

- Qwen3-8B without thinking: 153 question × variant pairs, 207 runs that took the bug vs 285 correct. Qwen3-8B with thinking takes the bug in only 1–2% of runs, too few to use.
- Qwen3-14B with thinking: 91 pairs, 129 vs 230 runs.
- As a reference, runs that were told it was allowed to use the bug (82 and 136 runs).

**Control: the start of the model's turn.** Runs of the same prompt read identical text there, so any difference between the groups is zero by construction. Observed: 0.49 and 0.52.

| Score, averaged over positions | Qwen3-8B, no thinking | Qwen3-14B, thinking |
| --- | ---: | ---: |
| Eval score, whole reasoning (pre-registered) | 0.54 [0.50, 0.58] | 0.67 [0.61, 0.73] |
| Eval score, before the first mention of grader/bug/cheat/exploit | 0.51 [0.48, 0.54] | 0.52 [0.47, 0.57] |
| "Cheating" score, before the first mention (pre-registered) | 0.51 [0.47, 0.54] | 0.50 [0.45, 0.55] |
| "Wrong answer" score, before the first mention (pre-registered) | 0.53 [0.49, 0.56] | 0.45 [0.40, 0.51] |
| Text baseline: count of grader/bug/score/exploit words before the answer | 0.74 [0.70, 0.78] | 0.87 [0.83, 0.92] |

How to read: AUROC for "took the bug" vs "answered correctly", J-lens; the bootstrap resamples questions.

Over the whole reasoning, Qwen3-14B's runs that take the bug do score higher (0.67). But that difference sits after the reasoning starts discussing the grader, and the text itself separates the groups better (0.87). Before any mention, nothing separates them.

## Step 4: "cheating" and "wrong answer" concepts, and the trained hidden-hack model

Step 3's table has the grader-bug numbers. For the reward-hack organism ([hack_organism.md](hack_organism.md)) we used the warm-start model only, so the model is held fixed. Its runs that hack while the reasoning hides it (92) were compared with its runs that don't hack (71), on 59 problems, at reasoning positions before the code. Eval score 0.48 [0.39, 0.57], "cheating" 0.50 [0.41, 0.59], "wrong" 0.50 [0.40, 0.59]; the logit lens is the same. As in hack_organism.md, no readout sees the hidden hack before the code.

## Step 5: the answer position

At the position just before the final answer letter, all the scores separate the groups strongly:

| Score at the answer position | Qwen3-8B, no thinking | Qwen3-14B, thinking |
| --- | ---: | ---: |
| Eval score (" fictional", " simulation") | 0.92 [0.89, 0.95] (logit 0.90) | 0.86 [0.82, 0.90] (logit 0.80) |
| "Cheating" score | 0.89 [0.85, 0.92] (logit 0.88) | 0.80 [0.75, 0.85] (logit 0.92) |
| "Wrong answer" score | 0.81 [0.77, 0.85] | 0.27 [0.21, 0.32] (reversed) |
| Eval score, only runs whose text names the bug (exploratory) | 0.94 [0.92, 0.96] | 0.86 (all runs name it) |
| Eval score within 5 bands of lens confidence (exploratory) | 0.83–0.98 per band | 0.72–0.89 per band |
| Runs told the bug was allowed (mean eval score) | 3.81, between the two groups (4.59 / 3.24) | 2.61, like the takers (2.57 / 2.21) |

How to read: AUROC for "took the bug" vs "answered correctly", J-lens unless marked; the last row gives mean scores in nats.

These two checks rule out the obvious artefacts. The confidence bands check that the effect isn't just a flatter distribution: the J-lens's raw control-noun log-prob doesn't differ between the groups for Qwen3-8B (0.50). The text-names-the-bug subset checks that it isn't just whether the bug was mentioned. So at the moment of writing the gamed answer, the lens does carry " fictional", " simulation", " fake" and " cheat"-like content more than when the model writes the correct answer. That fits the paper's case studies, where " fake" shows up while Claude writes a fabrication. But it comes *after* the reasoning has said what it will do, and the logit lens finds it equally well. It is a readout of a decision already in the text, not a sign of hidden knowledge.

## What this means for the project

The paper's positive result rests on Claude-scale models and on transcripts where eval awareness is real and often voiced (71% verbalized in their blackmail runs). On Qwen3-8B the same readout picks up eval-flavoured *words and topics*. A prompt that says "benchmark" scores high, and so does a story request. A plain MMLU item scores low. So we can't use it as an off-the-shelf "the model knows this is a test" detector for our runs. Moving towards arguing against its own knowledge, the readouts add nothing before the text reveals the switch. That matches our earlier lens results (lens_main, instrument_q1q3, hack_organism).

## Caveats and deviations

- **Not their models.** Qwen3-8B is much smaller than Claude Sonnet 4.5 and Opus 4.6, and it rarely voices eval awareness. A model that is less aware of evaluations gives the score less to find. A failure here may be the model, not the method.
- **Not their data.** Their eval suites and deployment traffic are internal. Ours are public stand-ins. WildChat is heavy on fiction and role-play, which the paper warns inflates " fictional". OASST alone sits between the suites (3.07).
- **Our 100 control nouns** aren't theirs. The score's level depends on them; comparisons between groups don't much.
- **The "says it is a test" regex** has about 80% precision (16 of 20 read matches).
- **Two samples per prompt** at T 0.6. Generation was capped at 4,096 new tokens for the corpora: all 200 coding-task samples hit the cap, and 0–22 per suite elsewhere. Only reasoning-position scores use the completions.
- **Step 3 uses existing runs.** The Qwen3-8B runs are non-thinking, because the thinking model almost never takes the bug.
- **Fixes during the night.** A selection file with spaces in run IDs was mis-parsed, so the hack readout was re-run. The first ablation sanity check measured the wrong thing (it registered the hooks before the "before" pass). The partial ablation run (30 rollouts) was set aside and the run restarted after the fixed check showed the ablation works: eval tokens in the top 20 at 21 of 2,122 prompt positions before, 0 after. The scores at the start of the turn differ slightly between the two samples of the same prompt (median 0.01 nats, max 0.19), because each sample's pass covers a different total length. We use the mean of the two.

## Files

- Code: `scripts/gen/evalaware_corpus.py`, `scripts/lens/evalaware_tokens.py`, `scripts/lens/readout_evalaware.py`, `scripts/lens/evalaware_ablate.py`, `scripts/lens/evalaware_steer.py`, `scripts/lens/analyze_evalaware.py`, `scripts/explore/evalaware_select.py`, `scripts/explore/evalaware_chain.sh`; tokens `configs/evalaware_tokens.json`; the blackmail prompt `configs/explore/blackmail.json` (jlens's copy, checked by sha256).
- Raw runs and readouts (not in git): octavia `data/explore/evalaware/`, `data/readouts/evalaware/`; copies on arena9-cloud under `data/evalaware/`.
- Timing: about 3 h wall-clock (00:00–03:00) and about 1.3 GPU-hours on octavia: generation 44 min, readouts 25 min, ablation rollouts 30 min, steering check 20 min. The GPU was lent to another agent 01:00–01:25. No API cost.

