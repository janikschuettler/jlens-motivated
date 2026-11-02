# CLAUDE.md

## Project

A one-week interpretability project. Sometimes Qwen3-8B (thinking mode) switches to a hinted wrong answer and its chain of thought never mentions the hint ("motivated reasoning"). Does the Jacobian lens (J-lens) show the hidden driver, deception or conflict words, or only a preview of what the chain of thought says next?

- [docs/plan.md](docs/plan.md) is the research design: questions Q1–Q6, labels, word sets, metrics. Changes to it are the user's decision.
- [docs/roadmap.md](docs/roadmap.md) has the milestones (M0–M5 = Day 0–5), their done-criteria, the chosen thresholds and the go/no-go points.
- [docs/open-questions.md](docs/open-questions.md) parks the design questions until M0 is done. Each one says which milestone needs it settled.
- Status: planning. No code yet.

Core pieces:

- **Model:** `Qwen/Qwen3-8B`, the post-trained hybrid-thinking model (not `Qwen3-8B-Base`), in thinking mode. Pinned to revision `b968826d9c46dd6066d109eabc6255188de91218`. It has 36 blocks, d_model 4096, a vocabulary of 151,936 and an untied unembedding.
- **Lens:** Neuronpedia's J-lens for this model, from the Hugging Face repo `neuronpedia/jacobian-lens`, file `qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt`. Readout at layer ℓ: `unembed(h @ J[ℓ].T)`, where `unembed` is the final RMSNorm followed by the LM head. Workspace layers: blocks 14, 16, …, 32. The comparison is the logit lens (J = I). There is no J++ lens for Qwen3-8B.
- **Lens library:** `anthropics/jacobian-lens` (Python package `jlens`). Its `lens.apply` takes a text prompt, but teacher forcing must feed the saved token IDs (see below).
- **Data:** hinted multiple-choice questions (ARC-Challenge, MMLU) built with Mirtaheri & Belkin's repo (`seyedparsa/motivated-reasoning`, called "M&B" in these docs). Hint types: sycophancy, metadata, consistency. Changing their code is fine.
- **Key contrast:** motivated runs against resisted runs. Both have the hint in the prompt.

## Machines

| Machine | Role | Specs (checked 2026-11-02) |
| --- | --- | --- |
| arena9-cloud (this machine, runs the editor) | Editing, docs | No GPU, 15 GB RAM, about 46 GB free disk |
| arena9-mayor (`ssh arena9-mayor`) | All model work: generation, readouts, probes | RTX 4090 24 GB (compute 8.9, driver 595, CUDA 13.2), 64 CPUs, 503 GB RAM, 200 GB disk, Ubuntu 22.04 |

- The fleet list is the "arena-infra fleet" block in `~/.ssh/config`. The other `arena9-*` machines belong to other people; don't use them.
- 24 GB is enough for Qwen3-8B in bf16 (16.4 GB of weights), both for vLLM generation and for teacher-forced readout. It leaves vLLM only about 4–5 GB of KV cache (about 30k tokens at 144 KiB per token), which limits batch size. It can't hold larger models in bf16 (Qwen3-14B is 29.5 GB).
- **Not set up yet:**
  - Getting the repo onto mayor. The repo is private, and mayor's GitHub key is a deploy key for a different repo.
  - Backups. mayor's disk is the container's overlay filesystem, and whether it survives a container restart is unknown.

## Environments

uv manages both environments. Never use `pip install`. Generation and readout use separate envs because vLLM pins its own torch and transformers versions.

| Env | Path | Contains | Used for |
| --- | --- | --- | --- |
| `gen` | `envs/gen/` | vLLM, the M&B repo's dependencies | Sampling chains of thought, extracting answers, assigning labels |
| `lens` | `envs/lens/` | torch 2.11 and transformers 5.11 (the versions the lens was fit with), `jlens` | Teacher-forced readouts, word-set scores, probes, ablations |

- Each env is its own uv project with a `pyproject.toml` and a committed `uv.lock`. Python 3.12 in both.
- Run code with `uv run --project envs/gen …` or `uv run --project envs/lens …`. Add dependencies with `uv add --project envs/<env> …`.
- Create and lock the envs on mayor, where they run. mayor has uv 0.12.23 on PATH. This machine has uv 0.13.6 in `~/.local/bin`, which isn't on zsh's PATH.
- zsh here activates `/root/ARENA_materials/.venv` at startup. That venv belongs to another project, so don't use its Python. `uv run --project` ignores it.
- Never install a package into one env to make the other env's code run. A dependency conflict counts as a failure (rule 1).
- Both envs load the pinned model revision and the same tokenizer. Record the revision with every output.
- Model weights and lens files go to the Hugging Face cache (`~/.cache/huggingface` on mayor), outside the repo.

**The envs exchange files on disk, keyed by token IDs.**

- `gen` saves, for each run: prompt token IDs, completion token IDs, the log-probs of the sampled tokens, the decoded text, model and tokenizer revision, chat-template settings (`enable_thinking`), sampling parameters, seed, hint type and question ID.
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
- a word-set entry that isn't a single token
- teacher-forced log-probs that disagree with vLLM's beyond the roadmap's tolerance
- out-of-memory errors, missing files, dependency conflicts
- label counts or rates far from what the plan expects

These workarounds are off-limits unless the user asks for them: switching the model, lens or layers; falling back to re-tokenizing text; dropping or replacing word-set entries; skipping failing runs or catching errors and continuing; loosening a threshold or tolerance; truncating sequences; quantizing the model to make it fit; installing into the other env.

The go/no-go points in the roadmap are the user's call. Report the numbers and leave the decision to the user.

### 2. Never commit data or model weights

None of the following goes in git: model weights and the Hugging Face cache, lens matrices (downloaded or fitted), datasets, generated chains of thought and answers, labels, readout outputs, activations. `.gitignore` covers `data/` and the usual weight and data formats.

Git can't ignore files by size, so stage files by name (no `git add -A` or `git add .`) and check `git status` before every commit.

### 3. Never store full-vocabulary lens outputs

The vocabulary has 151,936 tokens, so a full distribution for every run × layer × position won't fit on disk. Compute the full softmax in memory, because the word-set log-probs need it. Then save only:

- the top-25 lens tokens (IDs and log-probs)
- the log-probs of the word-set tokens

This applies to every lens (J-lens, logit lens) and to every output, including debug dumps, notebook caches and intermediate files.

As a result, the word sets must be final and checked in the tokenizer before a readout run, including the per-question driver and truth tokens. Adding a word later means re-running the readout, not storing more output.
