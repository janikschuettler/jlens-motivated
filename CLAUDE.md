# CLAUDE.md

## Project

An interpretability project on **arguing against its own knowledge**. On questions Qwen3-8B (thinking mode) reliably answers correctly, a nudge sometimes makes it argue for a wrong option. Is the correct answer still represented internally while it does, and can the Jacobian lens (J-lens) detect this better than the chain of thought can?

- Prefer "arguing against its own knowledge" in write-ups. "Motivated reasoning" is fine as a colloquial shorthand (the user's call), but it implies a motive we can't establish, and a hint can be legitimate evidence, so don't build claims on it.
- "Own knowledge" is behavioural: an item is reliably known if at least 90% of unhinted samples answer it correctly.
- [docs/plan.md](docs/plan.md) is the research design (v1: behavioural pilot, one internal measurement, optional detection, plus a later list). Changes to it are the user's decision.
- The findings dashboard, [docs/findings/index.html](docs/findings/index.html) (built from [docs/findings/registry.json](docs/findings/registry.json)), is the running record of every experiment and finding so far: done, running and upcoming (see "Findings dashboard" below).
- [docs/roadmap.md](docs/roadmap.md) has the milestones (M0 tools; M1–M3 = the plan's Steps 1–3), their done-criteria, the chosen thresholds and the go/no-go points.
- [docs/open-questions.md](docs/open-questions.md) lists the undecided points, each with the milestone that needs it settled.
- Status: M0 done (all checks pass, [docs/reports/m0.md](docs/reports/m0.md)). Docs revised to the v1 design on 2026-11-02. M1 code is built and smoke-tested (`scripts/gen/pilot.py`); the full pilot waits for the temperature decision (open question 1). Exploration on 2026-11-02/03 (M&B gap factorial, hint families, lens readouts, pre-registered lens tests, subtle cues, planted claims) is in [docs/overnight/](docs/overnight/); start from its log and report.

## Layout

```
pyproject.toml        jmr package (light deps only; installed into both envs)
envs/gen/, envs/lens/ one uv project per env, each with a committed uv.lock (envs/judge/: OpenRouter judge, CPU only)
configs/              model, lens, sampling, conditions (the nudges), data (YAML)
src/jmr/              shared code: config, records, tokens, anchors, prompts, questions, label, checks
src/jmr/gen/          gen env only (imports vllm)
src/jmr/lens/         lens env only (imports torch, jlens)
src/jmr/analysis/     analysis of stored samples and readouts (M1–M3)
scripts/gen/, scripts/lens/   thin CLI entry points; the folder says which env runs them
tests/                CPU tests (pytest)
data/                 gitignored: runs/<run_set>/ (items.jsonl, runs.jsonl, labels.jsonl, audit.md, pilot_summary.md), m0/, logs/
```

- The top level of `jmr` never imports torch or vllm. `jmr.gen` imports vllm, and `jmr.lens` imports torch and jlens.
- Every labelled run set gets `audit.md` (`jmr.audit`, written by `scripts/gen/label.py`): a summary table plus random examples per label × condition, with full prompts, complete chains of thought and responses, for auditing by eye. Never truncate chains of thought in it. `scripts/pull_audit.sh <run_set>` copies it (and `pilot_summary.md`) from the GPU machine to this machine.
- Prompts, conditions and labels are our own code (`jmr.prompts`, `jmr.label`). No third-party pipeline is used or ported.

Core pieces:

- **Model:** `Qwen/Qwen3-8B`, the post-trained hybrid-thinking model (not `Qwen3-8B-Base`), in thinking mode. Pinned to revision `b968826d9c46dd6066d109eabc6255188de91218`. It has 36 blocks, d_model 4096, a vocabulary of 151,936 and an untied unembedding.
- **Lens:** Neuronpedia's J-lens for this model, from the Hugging Face repo `neuronpedia/jacobian-lens`, file `qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt`. Readout at layer ℓ: `unembed(h @ J[ℓ].T)`, where `unembed` is the final RMSNorm followed by the LM head. Workspace layers: blocks 14, 16, …, 32. The comparison is the logit lens (J = I). There is no J++ lens for Qwen3-8B.
- **Lens library:** `anthropics/jacobian-lens` (Python package `jlens`). Its `lens.apply` takes a text prompt, but teacher forcing must feed the saved token IDs (see below).
- **Data:** multiple-choice questions from MMLU-Redux 2.0 (`error_type == "ok"` only). This is the only dataset to use now, including for debugging. ARC-Challenge and raw MMLU stay in the config for later (`status: later`, refused unless `allow_later=True`). 10 samples per prompt. Five conditions: unhinted, null control, zero-evidence nudge, known-evidence nudge, positive control (instructed to argue for B).
- **Key contrast:** on reliably known items, nudged samples that followed the hinted wrong option B against nudged samples that resisted, with instructed samples as a reference. A "silent switch" is a followed sample whose chain of thought doesn't mention the hint.

## Machines

| Machine | Role | Specs (checked 2026-11-02) |
| --- | --- | --- |
| arena9-cloud (this machine, runs the editor) | Editing, docs | No GPU, 15 GB RAM, about 46 GB free disk |
| arena9-octavia (`ssh arena9-octavia`) | All model work now: generation, readouts, probes | NVIDIA L40S 46 GB (48 GB marketed), driver 580, CUDA 13.0, 192 CPUs, about 2 TB RAM, 100 GB disk (96 GB free before setup), vast.ai container (checked 2026-11-02) |
| arena9-derpy (`ssh arena9-derpy`) | Second GPU machine for model work (the user's decision, 2026-11-04, for the rest of the project) | NVIDIA L40S 46 GB, 96 CPUs, about 1 TB RAM, 300 GB disk (297 GB free), vast.ai container (checked 2026-11-04). Has more disk than octavia: put large weights and readouts here |

- The fleet list is the "arena-infra fleet" block in `~/.ssh/config`. The other `arena9-*` machines belong to other people; don't use them.
- The L40S has about 45 GiB usable. Qwen3-8B in bf16 is 16.4 GB, which leaves roughly 24 GB for vLLM's KV cache (the 24 GB card used for M0 and the pilots had about 4 GB). That allows far more concurrent sequences, so generation should be several times faster than in M0. It still can't hold Qwen3-32B in bf16 (about 65 GB); Qwen3-14B (about 30 GB) fits for readouts. Disk is 100 GB, so watch `data/` and the HF cache.
- **Code gets to the GPU machine via `scripts/sync_to_gpu.sh`** (an rsync of the repo, excluding `data/`, `third_party/` and venvs; it pulls newer `uv.lock` files back first). The target defaults to `arena9-octavia`; set `GPU_HOST=arena9-derpy` (or another ssh host) to use another machine (same for `scripts/pull_audit.sh <run_set>`). Edit here, sync, then run with `ssh arena9-octavia 'cd /root/jlens-motivated && unset VIRTUAL_ENV && uv run --project envs/<env> python scripts/<env>/<script>.py'`. The repo is private and no GPU machine has a key that can `git pull` it.
- **Not set up yet:** backups. Octavia is a vast.ai container, and whether it survives a restart is unknown. `data/` (78 MB, expensive to regenerate) exists only there. Model weights and the lens can be re-downloaded.

## Environments

uv manages both environments. Never use `pip install`. Generation and readout use separate envs because vLLM pins its own torch and transformers versions.

| Env | Path | Contains | Used for |
| --- | --- | --- | --- |
| `gen` | `envs/gen/` | vLLM 0.33, torch 2.14, transformers 5.20, datasets | Sampling chains of thought, extracting answers, assigning labels |
| `lens` | `envs/lens/` | torch 2.11 and transformers 5.11 (the versions the lens was fit with), `jlens` pinned to a commit | Teacher-forced readouts of tracked option tokens |

- Each env is its own uv project with a `pyproject.toml` and a committed `uv.lock`. Python 3.12 in both. Each depends on `jmr` as an editable path dependency.
- The two envs have different transformers versions, so the tokenizer fingerprint (`jmr.tokens.fingerprint`) is checked across envs (M0.4).
- Run code with `uv run --project envs/gen …` or `uv run --project envs/lens …`. Add dependencies with `uv add --project envs/<env> …`.
- Create and lock the envs on the GPU machine, where they run (`uv sync --project envs/<env>`). Both machines have uv 0.12.23 on PATH. This machine has uv 0.13.6 in `~/.local/bin`, which isn't on zsh's PATH.
- zsh here activates `/root/ARENA_materials/.venv` at startup. That venv belongs to another project, so don't use its Python. `uv run --project` ignores it.
- Never install a package into one env to make the other env's code run. A dependency conflict counts as a failure (rule 1).
- Both envs load the pinned model revision and the same tokenizer. Record the revision with every output.
- Model weights and lens files go to the Hugging Face cache (`~/.cache/huggingface` on the GPU machine), outside the repo. Download them there directly (`huggingface_hub`); don't copy the cache between machines (6–7 MB/s, versus about 90 s for a direct download).

**The envs exchange files on disk, keyed by token IDs.**

- `gen` saves, for each run: prompt token IDs, completion token IDs, the log-probs of the sampled tokens, the decoded text, model and tokenizer revision, chat-template settings (`enable_thinking`), sampling parameters, seed, condition, item ID and hinted option.
- `lens` teacher-forces exactly the saved IDs. It never re-tokenizes decoded text or re-applies the chat template, because re-tokenizing text doesn't reliably reproduce the generated IDs. If `jlens` can't take token IDs, apply the J matrices directly and check the result against `jlens` (roadmap M0.2).
- **Exception (the user's decision, 2026-11-04): runs generated through an API** (which returns text, not token IDs) may be re-tokenized with the pinned tokenizer and chat template, so that larger models can be read out locally. Conditions: a fidelity check on every such run (its per-token log-probs under our local model, compared with the model's own generations; off-policy runs flagged by a rule fixed in advance), every such run labelled "re-tokenized from API text" in data and reports, and a note that the provider may serve different weights. Runs we generate ourselves still use their saved IDs.
- The token IDs are the record of what was generated. Text from vLLM's reasoning parser is not.
- Run data lives in `data/` in the repo, which is gitignored.

## Working style

- For minor parameters and thresholds, pick a reasonable default, write it down (in the roadmap or the code) and tell the user. Design questions that don't block the current milestone go into `docs/open-questions.md`; don't stop on them.
- Thresholds already in the roadmap are not to be loosened without asking (rule 1).

### Writing

- Use simple, concise prose: short sentences and plain words. Use jargon only where it's needed, and say what it means the first time.
- Give every new finding an interpretation line: what it means for the project's question, and how sure we are. This matters most in summary artefacts (reports, the findings dashboard, web pages).

**Experiment summaries must be readable by someone who didn't run the experiment** (the user's instruction, 2026-11-03). This applies to the dashboard, reports' summaries, and chat messages to the user. The test: the reader can explain the result back after one read, without asking what a label or number means.

- **Lead with the plain answer** to the question the experiment asked ("No: the readouts can't tell a hidden hack from a clean run"), then the evidence, then caveats.
- **No internal labels without a plain description.** Don't write H1, T1b, V2′, C3, I3, β 0.02, "warm-start", arm or cell names on their own. Say what the thing is ("the version of the model trained to hide its hack", "the pre-registered main test"). If a label is needed for cross-reference, put it in brackets after the description.
- **Every number says what was compared and what it measures.** Not "AUROC 0.995", but "the score separates hacking runs from clean runs almost perfectly (AUROC 0.995, where 0.5 is chance and 1 is perfect)". Units and scales on first use (nats, percentage points, AUROC).
- **Every control gets one sentence on what it checks and why** ("we also compared the two models on identical text: if the score still separates them, it's detecting the model, not the hack").
- **Short sentences, at most two or three numbers per sentence.** No chains of parenthetical numbers. Put dense numbers in a table instead.
- **Tables:** plain-language column headers and row labels that make sense on their own, plus one line under the table on how to read it.
- **Define a condition or group by what happens in it** ("the model was told to argue for the wrong answer"), not by the name of its config.

### Findings dashboard

The running record is a web page, [docs/findings/index.html](docs/findings/index.html), built from one source of truth, [docs/findings/registry.json](docs/findings/registry.json). It is published as a private claude.ai artifact: https://claude.ai/artifact/BZHUWiBPLb7zN7mzXXbvzy (slug `557d5704-e0b7-4329-8053-f7b84ab11dc6`). Republish to that slug (never create a new artifact for it), reading the live version first for `base_version`. `docs/findings.md` is only a pointer; don't write findings there.

- **Registry:** `thread` holds the main question, a summary, and the strands (sub-questions A–C, tools T, beyond hinted questions N). Each strand has a hypothesis, a current answer with a confidence, and a next step. `experiments` has one entry per experiment, done, running or upcoming. Each entry has: id, title, strand, status, kind (exploratory, pre-registered, definitive or tooling), verdict against its strand's hypothesis, question, method (short but exact), result, key numbers, interpretation line, confidence, caveats, n, dates, timing (wall-clock, GPU-hours, API cost), machine, figure, reports, derived data, commits and related entries. The schema is checked by the build script, which lists every allowed value. Use derived numbers only.
- **After every result** (and when an experiment starts or is planned), in the same commit as the result's report:
  1. update the entry in `registry.json`: status, numbers, interpretation, verdict, timing, commits. Update the strand's current answer and the thread summary if the result changes them. Update `meta.updated`.
  2. if the result has a figure, add or redraw it with `scripts/docs/make_figures.py` (SVGs in `docs/findings/figs/`, from derived numbers only) and set the entry's `figure`.
  3. run `~/.local/bin/uv run --project envs/judge python scripts/docs/build_dashboard.py`. It fails loudly on a missing or malformed field; fix the registry, don't loosen the check.
  4. commit the registry, figures and `docs/findings/index.html`. The orchestrator republishes the artifact.
- **Style:** simple prose, short and exact. Every finding gets an interpretation line. Mark each as exploratory or pre-registered (the `kind` field). When a result is superseded, update its entry in place; don't stack old and new versions.
- **Page code:** `scripts/docs/dashboard_template.html` (CSS and JS, inline; views `#thread`, `#experiments`, `#terms`, `#exp-<id>`). Keep it one self-contained document with no remote scripts or images.

### Commits

Commit on your own, without asking, at natural checkpoints (the user's standing instruction, 2026-11-03):

- **When:** a piece of code works and its tests pass; an experiment's report is written; before launching a long GPU run, so the run records a clean git state; at the end of a working session. Don't commit failing tests or half-finished code.
- **Granularity:** one logical change per commit (a tool, an experiment's code + config, a report), not one commit per file and not a day's work in one commit.
- **Message:** a short imperative subject naming what changed, then a body with what and why (key numbers for an experiment, the report's path), ending with the attribution line.
- **Checks before every commit:** stage files by name and check `git status` and `git diff --cached --stat` (rule 2); nothing under `data/`, no weights, no generated chains of thought beyond short quotes in reports; scan staged files for API keys and tokens.
- **Branch and remote:** commit on the current branch (`main` so far). Don't push, amend or rewrite history unless the user asks.

## Rules

### 1. Stop and report on failures; don't work around them

When something fails or doesn't match expectations, stop, report and wait. A report gives the command, the exact error or numbers, what you think it means and the options you see.

This applies especially to:

- the lens's shape, layer count or capture point not matching the checkpoint
- teacher-forcing token IDs that differ from the generated ones
- a tracked token that isn't a single token
- teacher-forced log-probs that disagree with vLLM's beyond the roadmap's tolerance
- out-of-memory errors, missing files, dependency conflicts
- sample counts, parse rates or label rates far from what the plan expects

These workarounds are off-limits unless the user asks for them: switching the model, lens or layers; falling back to re-tokenizing text; dropping or replacing tracked tokens; skipping failing runs or catching errors and continuing; loosening a threshold or tolerance; truncating sequences; quantizing the model to make it fit; installing into the other env.

The go/no-go points in the roadmap are the user's call. Report the numbers and leave the decision to the user.

### 2. Never commit data or model weights

None of the following goes in git: model weights and the Hugging Face cache, lens matrices (downloaded or fitted), datasets, generated chains of thought and answers, labels, readout outputs, activations. `.gitignore` covers `data/` and the usual weight and data formats.

Git can't ignore files by size, so stage files by name (no `git add -A` or `git add .`) and check `git status` before every commit.

### 3. Never store full-vocabulary lens outputs

The vocabulary has 151,936 tokens, so a full distribution for every sample × layer × position won't fit on disk. Compute the full softmax in memory, because ranks and log-probs need it. Then save only:

- the top-25 lens tokens (IDs and log-probs)
- the rank and log-prob of the tracked tokens (the option tokens: correct, hinted, other wrong)

This applies to every lens (J-lens, logit lens) and to every output, including debug dumps, notebook caches and intermediate files.

As a result, the tracked tokens must be final and checked in the tokenizer before a readout run. Adding a token later means re-running the readout, not storing more output.
