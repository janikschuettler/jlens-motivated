# CLAUDE.md

## Project

An interpretability project on **arguing against its own knowledge**. On questions Qwen3-8B (thinking mode) reliably answers correctly, a nudge sometimes makes it argue for a wrong option. Is the correct answer still represented internally while it does, and can the Jacobian lens (J-lens) detect this better than the chain of thought can?

- Prefer "arguing against its own knowledge" in write-ups. "Motivated reasoning" is fine as a colloquial shorthand (the user's call), but it implies a motive we can't establish, and a hint can be legitimate evidence, so don't build claims on it.
- "Own knowledge" is behavioural: an item is reliably known if at least 90% of unhinted samples answer it correctly.
- [docs/plan.md](docs/plan.md) is the research design (v1: behavioural pilot, one internal measurement, optional detection, plus a later list). Changes to it are the user's decision.
- [docs/roadmap.md](docs/roadmap.md) has the milestones (M0 tools; M1–M3 = the plan's Steps 1–3), their done-criteria, the chosen thresholds and the go/no-go points.
- [docs/open-questions.md](docs/open-questions.md) lists the undecided points, each with the milestone that needs it settled.
- Status: M0 done (all checks pass, [docs/reports/m0.md](docs/reports/m0.md)). Docs revised to the v1 design on 2026-11-02. M1 code is built and smoke-tested (`scripts/gen/pilot.py`); the full pilot waits for the temperature decision (open question 1).

## Layout

```
pyproject.toml        jmr package (light deps only; installed into both envs)
envs/gen/, envs/lens/ one uv project per env, each with a committed uv.lock
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
- Every labelled run set gets `audit.md` (`jmr.audit`, written by `scripts/gen/label.py`): a summary table plus random examples per label × condition, with full prompts, complete chains of thought and responses, for auditing by eye. Never truncate chains of thought in it. `scripts/pull_audit.sh <run_set>` copies it (and `pilot_summary.md`) from mayor to this machine.
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
| arena9-mayor (`ssh arena9-mayor`) | All model work: generation, readouts, probes | RTX 4090 24 GB (compute 8.9, driver 595, CUDA 13.2), 64 CPUs, 503 GB RAM, 200 GB disk, Ubuntu 22.04 |

- The fleet list is the "arena-infra fleet" block in `~/.ssh/config`. The other `arena9-*` machines belong to other people; don't use them.
- 24 GB is enough for Qwen3-8B in bf16 (16.4 GB of weights), both for vLLM generation and for teacher-forced readout. It leaves vLLM only about 4–5 GB of KV cache (about 30k tokens at 144 KiB per token), which limits batch size. It can't hold larger models in bf16 (Qwen3-14B is 29.5 GB).
- **Code gets to mayor via `scripts/sync_to_mayor.sh`** (an rsync of the repo, excluding `data/` and venvs; it pulls newer `uv.lock` files back first). Edit here, sync, then run on mayor with `ssh arena9-mayor 'cd /root/jlens-motivated && uv run --project envs/<env> python scripts/<env>/<script>.py'`. The repo is private and mayor's GitHub key belongs to a different repo, so mayor can't `git pull`.
- **Not set up yet:** backups. mayor's disk is the container's overlay filesystem, and whether it survives a container restart is unknown.

## Environments

uv manages both environments. Never use `pip install`. Generation and readout use separate envs because vLLM pins its own torch and transformers versions.

| Env | Path | Contains | Used for |
| --- | --- | --- | --- |
| `gen` | `envs/gen/` | vLLM 0.33, torch 2.14, transformers 5.20, datasets | Sampling chains of thought, extracting answers, assigning labels |
| `lens` | `envs/lens/` | torch 2.11 and transformers 5.11 (the versions the lens was fit with), `jlens` pinned to a commit | Teacher-forced readouts of tracked option tokens |

- Each env is its own uv project with a `pyproject.toml` and a committed `uv.lock`. Python 3.12 in both. Each depends on `jmr` as an editable path dependency.
- The two envs have different transformers versions, so the tokenizer fingerprint (`jmr.tokens.fingerprint`) is checked across envs (M0.4).
- Run code with `uv run --project envs/gen …` or `uv run --project envs/lens …`. Add dependencies with `uv add --project envs/<env> …`.
- Create and lock the envs on mayor, where they run. mayor has uv 0.12.23 on PATH. This machine has uv 0.13.6 in `~/.local/bin`, which isn't on zsh's PATH.
- zsh here activates `/root/ARENA_materials/.venv` at startup. That venv belongs to another project, so don't use its Python. `uv run --project` ignores it.
- Never install a package into one env to make the other env's code run. A dependency conflict counts as a failure (rule 1).
- Both envs load the pinned model revision and the same tokenizer. Record the revision with every output.
- Model weights and lens files go to the Hugging Face cache (`~/.cache/huggingface` on mayor), outside the repo.

**The envs exchange files on disk, keyed by token IDs.**

- `gen` saves, for each run: prompt token IDs, completion token IDs, the log-probs of the sampled tokens, the decoded text, model and tokenizer revision, chat-template settings (`enable_thinking`), sampling parameters, seed, condition, item ID and hinted option.
- `lens` teacher-forces exactly the saved IDs. It never re-tokenizes decoded text or re-applies the chat template, because re-tokenizing text doesn't reliably reproduce the generated IDs. If `jlens` can't take token IDs, apply the J matrices directly and check the result against `jlens` (roadmap M0.2).
- The token IDs are the record of what was generated. Text from vLLM's reasoning parser is not.
- Run data lives in `data/` in the repo, which is gitignored.

## Working style

- For minor parameters and thresholds, pick a reasonable default, write it down (in the roadmap or the code) and tell the user. Design questions that don't block the current milestone go into `docs/open-questions.md`; don't stop on them.
- Thresholds already in the roadmap are not to be loosened without asking (rule 1).

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
