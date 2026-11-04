# Does the J-lens eval-awareness result replicate, and does it carry over to arguing against its own knowledge?

RESULT_PLACEHOLDER

## Step 0: what the paper did

**Source.** Gurnee, Lindsey et al. 2026, "Verbalizable Representations Form a Global Workspace in Language Models", Transformer Circuits Thread ([transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html)). It is the paper that introduces the J-lens; the `jlens` package we use (`anthropics/jacobian-lens`, commit 581d3986) is its companion code. The eval-awareness results are in the section "Using the J-lens for alignment auditing" and the appendix "Measuring evaluation awareness with the J-lens". Credits in the paper: blackmail case study Jack Lindsey and Wes Gurnee; "Measuring evaluation awareness" Jack Lindsey and Subhash Kantamneni.

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

STEPS_PLACEHOLDER
