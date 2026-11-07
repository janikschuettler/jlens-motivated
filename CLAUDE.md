# CLAUDE.md

## Project

Interpretability project on **arguing against its own knowledge**. On questions a model reliably answers correctly, a nudge sometimes makes it argue for a wrong option. Is the correct answer still represented internally while it does, and can the Jacobian lens (J-lens) see this better than the chain of thought (CoT)?

- Prefer "arguing against its own knowledge" in write-ups. "Motivated reasoning" is OK colloquially (the user's call), but it implies a motive we can't establish; don't build claims on it.
- "Own knowledge" is behavioural: an item is reliably known if ≥ 90% (≥ 9/10) of unhinted samples answer it correctly. Lighter filters (5/5) must be labelled as such.
- [docs/plan.md](docs/plan.md) is the research design (v1); [docs/roadmap.md](docs/roadmap.md) milestones M0–M3, thresholds, go/no-go points; [docs/open-questions.md](docs/open-questions.md) undecided points. Plan changes and go/no-go calls are the user's decisions.
- **Source of truth for findings:** `docs/findings.md` (curated current state), `docs/log.jsonl` (chronological record) and one folder per experiment in `docs/experiments/`. The dashboard `docs/dashboard/index.html` is built from them (see "Findings dashboard").

**Working memory (the user's instruction, 2026-11-05).** Chats get cleared regularly. A new chat must be able to start from this file plus [docs/findings.md](docs/findings.md).
- `docs/findings.md` is the curated current state: short claims, each naming its supporting experiments by folder id, plus what's ruled out and the open questions. **Rewrite it after every result, every decision and before ending a session.** Rewrite claims in place; don't append. Keep it under about 60 lines.
- `docs/log.jsonl` is the chronological record, **append-only**: one JSON object per line, `{"ts", "type", "exp", "text", "by"}`. Types: decision, start, result, dead_end, surprise, note; `exp` is a list of experiment ids; `by` is `user` or `claude`. Log every decision (above all the user's), start, result, dead end and surprise in one or two sentences, linking files for detail. Never edit or reorder old lines.
- Each experiment has a folder `docs/experiments/<id>/` with a `README.md` (YAML front matter with the short structured fields, then Markdown sections Question, Method, Result, Interpretation, Caveats, Files), its `report.md` if any, its figures (`*.svg`) and its derived numbers (`data/*.json`). Create the folder when the experiment starts (status `running` or `upcoming`) and update the README when it finishes.
- For details, grep (don't read in full) `docs/log.jsonl` and `docs/experiments/*/README.md`.

**Credit use (the user's instruction, 2026-11-05).**
- Opus runs the main thread only. Subagents default to Sonnet (`CLAUDE_CODE_SUBAGENT_MODEL=sonnet` in the profile settings). Request Opus for a subagent only when the task really needs it, and say why.
- Spawn subagents only for independent parallel work or large reading jobs, never for lookups you can do in a few calls. Give them a narrow brief that points at `docs/findings.md` and the exact files.
- Prefer cheap calls: `grep`, `head`, `sed -n` and line-range reads over reading whole files or transcripts; local scripts over API judge calls; small smoke runs before full GPU runs. Tell subagents to do the same. Don't poll; use background jobs and notifications.

**Current hypothesis and status (2026-11-05).** M0 done ([docs/experiments/m0-tools/report.md](docs/experiments/m0-tools/report.md)); M1 pilot code built, full pilot never run (superseded by exploration, see `docs/log.jsonl`). Working conclusion: models can be made to argue against their knowledge (explicit or planted answer claims; other model families when told they're scored), but in thinking mode they almost always say so in the CoT, and the J-lens has not shown anything the CoT, the logit lens or a trained probe doesn't. One narrow pre-registered J-lens pass (planted claim, answer position, AUROC 0.71) is not yet robust. Claim strengths: [docs/redteam_claims.md](docs/redteam_claims.md). Slide deck source: [docs/presentation/slides.md](docs/presentation/slides.md).

## Layout

```
pyproject.toml        jmr package (light deps only; installed into all envs)
envs/gen/, envs/lens/ one uv project per env, committed uv.lock; envs/judge/: OpenRouter judge + docs scripts, CPU only
configs/              model, lens, sampling, conditions, data (YAML); configs/explore/*.yaml experiment specs
src/jmr/              shared code: config, records, tokens, anchors, prompts, questions, label, checks, judge, moral
src/jmr/gen/          gen env only (imports vllm)
src/jmr/lens/         lens env only (imports torch, jlens); batch.py = batch readout
src/jmr/explore/      exploration runner (arm specs, multi-turn, prefill, forks); spec format in __init__ docstring
src/jmr/analysis/     analysis of stored samples and readouts
scripts/gen|lens|explore|docs/   thin CLIs; the folder says which env runs them
tests/                CPU tests (pytest)
docs/findings.md      curated current state; docs/log.jsonl: append-only chronological log
docs/experiments/<id>/ README.md (front matter + details), report.md, figures, data/*.json (derived numbers)
docs/dashboard/       site.yaml (main question, strands, terms) and the built index.html
docs/                 also preregistration.md, redteam_claims.md, design_probe_control.md, plan.md, roadmap.md, open-questions.md, presentation/; unsorted/ holds files not yet assigned to an experiment
docs/literature/      literature notes
data/                 gitignored run data (top-level only; docs/experiments/*/data/ is tracked)
```

- Top-level `jmr` never imports torch or vllm. `jmr.gen` imports vllm; `jmr.lens` imports torch and jlens.
- Every labelled run set gets `audit.md` (`jmr.audit`): summary table plus random examples per label × condition with full prompts, complete CoTs and responses. Never truncate CoTs in it. `scripts/pull_audit.sh <run_set>` copies it here.
- Prompts, conditions and labels are our own code (`jmr.prompts`, `jmr.label`). No third-party pipeline is used or ported (M&B prompt strings are rebuilt in our code).

**Core pieces.**
- **Model:** `Qwen/Qwen3-8B` (post-trained hybrid-thinking, not Base), thinking mode, revision `b968826d9c46dd6066d109eabc6255188de91218`; 36 blocks, d_model 4096, vocab 151,936, untied unembedding. Other models used in exploration: Qwen3-14B, Qwen3.6-27B, gpt-oss-20b, `cds-jb` reward-hack LoRAs on Qwen3-8B; pin revisions in each report.
- **Lens:** Neuronpedia J-lens, HF `neuronpedia/jacobian-lens` (revision `e166deb5…`), file `qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt`. Readout at layer ℓ: `unembed(h @ J[ℓ].T)` (final RMSNorm + LM head). Workspace layers 14, 16, …, 32. Comparison: logit lens (J = I). No J++ for Qwen3-8B (J++ exists for Qwen3.6-27B and others).
- **Lens library:** `anthropics/jacobian-lens` (`jlens`, pinned commit 581d3986). Teacher forcing feeds saved token IDs.
- **Data:** MMLU-Redux 2.0, `error_type == "ok"` only. ARC-Challenge and raw MMLU stay `status: later` (refused unless `allow_later=True`). Exploration beyond Redux (MoralChoice, coding tasks, blackmail/eval-awareness prompts) was user-approved; label it in reports.
- **Key contrast:** on known items, nudged runs that followed the hinted wrong option B vs runs that resisted, with instructed runs as reference. A "silent switch" is a followed run whose CoT doesn't mention the hint.

## Machines

| Machine | Role | Specs |
| --- | --- | --- |
| arena9-cloud (this machine) | Editing, docs, CPU analysis, archive (`data/octavia_archive/`, `data/hack/`) | No GPU, 15 GB RAM, ~25 GB free disk |
| arena9-octavia | GPU work | L40S 46 GB usable, 192 CPUs, ~2 TB RAM, 100 GB disk (often ~13 GB free; Qwen3.6-27B weights in `/dev/shm/hf`) |
| arena9-derpy | Second GPU machine (user's decision 2026-11-04, for the rest of the project) | L40S 46 GB, 96 CPUs, ~1 TB RAM, 300 GB disk; put large weights/readouts here. Envs built 2026-11-04; Qwen3-8B/14B downloads were started |

- Fleet list: "arena-infra fleet" block in `~/.ssh/config`. Other `arena9-*` machines belong to others; don't use them. **arena9-mayor** only on nights the user explicitly allows it; never in standing docs.
- If GPU size or count is the bottleneck, ask the user first: they can request 96/128 GB or extra 24/48 GB GPUs. Qwen3-32B/27B don't fit in bf16; use CPU offload (exact, ~5 s per readout run), never quantization.
- One agent owns one GPU at a time; check `nvidia-smi` and processes before starting GPU jobs. Keep ≥ 10 GB free disk; move bulky readouts to derpy or arena9-cloud with checksum verification (sort with `LC_ALL=C`) before deleting.
- Code reaches GPUs via `scripts/sync_to_gpu.sh` (rsync `--delete`; excludes `/data/`, `/third_party/`, venvs; pulls newer `uv.lock` back first). Default target octavia; `GPU_HOST=arena9-derpy` for derpy. Edit here, sync, then `ssh <host> 'cd /root/jlens-motivated && unset VIRTUAL_ENV && uv run --project envs/<env> python scripts/<dir>/<script>.py'`. Never edit code only on a GPU machine (sync deletes it). The repo is private; GPU machines can't `git pull`. Use nohup + log files for long jobs.
- No automatic backups; vast.ai containers may not survive restarts. **Backup of run data (2026-11-07):** private HF dataset `janiiiks/jlens-motivated-data` holds all generations, labels and logs from `data/` on octavia (`octavia/`) and derpy (`derpy/`), plus per-run labels (`cloud/docs/`) and the generations that existed only on arena9-cloud (`cloud/data/`: API cross-model runs, grader runs, judge cache, manual-read sheets; own `cloud/SHA256SUMS_data`), with `SHA256SUMS`; lens readouts and weights are not in it (recompute / re-download). Needs a write token on the user's account to add to it.

## Environments

uv only; never `pip install`. Separate envs because vLLM pins its own torch/transformers.

| Env | Contains | Used for |
| --- | --- | --- |
| `gen` | vLLM 0.33, torch 2.14, transformers 5.20, datasets | Sampling CoTs, answers, labels |
| `lens` | torch 2.11, transformers 5.11 (versions the lens was fit with), `jlens` | Teacher-forced readouts, probes, steering (HF) |
| `judge` | jmr + httpx | OpenRouter calls; `scripts/docs/*` (stdlib) |

- Python 3.12; each env depends on `jmr` editable; committed `uv.lock`. Create/lock on the GPU machine (`uv sync --project envs/<env> --frozen`). uv 0.12.23 on GPU machines; here `~/.local/bin/uv` (0.13.6, not on zsh PATH).
- zsh here activates `/root/ARENA_materials/.venv` (another project): don't use its Python here; `uv run --project` ignores it.
- Tokenizer fingerprint (`jmr.tokens.fingerprint`) is checked across envs. Never install into one env to run the other's code; a dependency conflict is a rule-1 failure.
- Weights and lens files go to the HF cache on the GPU machine (or `/dev/shm` for big models), downloaded directly; don't copy caches between machines.
- **OpenRouter:** read `OPENROUTER_API_KEY` only in a fresh shell (`zsh -ic '…'`); the copy inherited by agent processes was stale. Never print or store the key. Approved judge models: gemini-2.5-flash-lite, gpt-5-nano. Key had ~$1–3 left on 2026-11-05: set a cap per task and ask before larger spend.

**Envs exchange files keyed by token IDs.**
- `gen` saves per run: prompt and completion token IDs, sampled-token log-probs (`raw_logprobs`), decoded text, model/tokenizer revision, `enable_thinking`, sampling params, seed, condition, item ID, hinted option (+ arm spec for exploration).
- `lens` teacher-forces exactly the saved IDs; never re-tokenizes decoded text or re-applies the chat template.
- **Exception (user, 2026-11-04): API-generated runs** may be re-tokenized with the pinned tokenizer and template, with a fidelity check on every run (per-token log-probs under our model vs its own generations; off-policy runs flagged by a rule fixed in advance), the label "re-tokenized from API text", and a note that providers may serve different weights.
- Token IDs are the record of what was generated, not vLLM's reasoning-parser text. Run data lives in gitignored `data/`.

## Working style

- Minor parameters: pick a reasonable default, write it down, tell the user. Non-blocking design questions go to `docs/open-questions.md`. Roadmap thresholds are not loosened without asking.
- Pre-register (timestamped, in `docs/preregistration.md`) the score, groups, prediction and fail/underpowered rule before reading out a test; label everything else exploratory. Count items, not runs, for CIs (item-level bootstrap).
- Controls to include by default: salience control (an unchosen but prominent option's margin), echo control (read before the text names the thing), same-text-different-model, within-model comparisons, text/length baselines, logit lens next to J-lens.

### Writing (user's instructions, 2026-11-03)
- Simple, concise prose; jargon only where needed, defined on first use. Every finding gets an interpretation line (what it means, how sure).
- Summaries (dashboard, report summaries, chat) must be readable by someone who didn't run the experiment: lead with the plain answer; no internal labels (H1, T1b, V2′, C3, β 0.02, "warm-start", cell names) without a plain description; every number says what was compared and what it measures (e.g. "AUROC 0.71, where 0.5 is chance"); every control gets one sentence on what it checks; ≤ 2–3 numbers per sentence; tables with plain headers and a how-to-read line; define conditions by what happens in them.

### Findings dashboard
- `docs/dashboard/index.html` is built by `~/.local/bin/uv run --project envs/judge python scripts/docs/build_dashboard.py` from `docs/dashboard/site.yaml` (main question, summary, strands A–C, T, N with hypothesis, current answer, confidence, next step; terms; meta), the experiment READMEs' front matter and sections, and `docs/log.jsonl`. The build checks the schema and fails loudly on a missing or malformed field; fix the source, don't loosen the check. Derived numbers only.
- Rebuild in batches (end of a working session, or when asked), not after every result. Before a rebuild: update `site.yaml` strand answers and `meta.updated` if findings changed, redraw figures with `scripts/docs/make_figures.py` (each SVG is written into its experiment folder), then commit sources, figures and `index.html` together.
- Published as private artifact https://claude.ai/artifact/BZHUWiBPLb7zN7mzXXbvzy (slug `557d5704-e0b7-4329-8053-f7b84ab11dc6`): republish to that slug only, reading the live version for `base_version`. Page code: `scripts/docs/dashboard_template.html`, one self-contained document.

### Commits (user's standing instruction, 2026-11-03)
- Commit without asking at checkpoints: working code with passing tests, a finished report, before a long GPU run, end of session. Never failing tests or half-finished code. One logical change per commit; imperative subject + body (what, why, key numbers, report path) + attribution line.
- Before every commit: stage by name, check `git status` and `git diff --cached --stat`; nothing from `data/`, no weights, no CoTs beyond short quotes; scan for keys. Commit on `main`; no push, amend or history rewrite unless asked.

## Rules

### 1. Stop and report on failures; don't work around them
On a failure or mismatch: stop, report (command, exact error or numbers, interpretation, options) and wait. Especially: lens shape/layer count/capture point mismatches; teacher-forced IDs differing from generated; a tracked token that isn't a single token; teacher-forced log-probs disagreeing with vLLM beyond tolerance; OOM, missing files, dependency conflicts; counts, parse or label rates far from expectation. Off-limits unless the user asks: switching model, lens or layers; re-tokenizing text (except the API exception above); dropping or replacing tracked tokens; skipping failing runs or swallowing errors; loosening thresholds; truncating sequences (lowering max_tokens counts: flag it and get acceptance); quantizing to fit; installing into the other env.

### 2. Never commit data or model weights
Not in git: weights, HF cache, lens matrices, datasets, generated CoTs and answers, labels, readout outputs, activations, per-run scores. `.gitignore` covers top-level `data/` (with an exception for `docs/experiments/` derived JSON) and weight/data formats. Stage files by name only.

### 3. Never store full-vocabulary lens outputs
Compute full softmaxes in memory; save only top-25 tokens (IDs, log-probs) at chosen positions and the rank/log-prob of tracked tokens (option letters, per-item content words, concept tokens), for every lens and every output incl. debug dumps. Tracked tokens must be final and checked before a readout; adding one means re-running. Open-vocabulary searches must aggregate in memory and store only the top few hundred differing tokens.
