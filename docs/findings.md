# Findings

The running record of the project now lives in the findings dashboard:

- **Page:** [findings/index.html](findings/index.html), one self-contained HTML file (also published as a claude.ai artifact). Open it in a browser. Views: `#thread` (the main question, sub-questions A–C plus tools and next strands, each with its current answer and experiments), `#experiments` (every experiment with status, kind, verdict, timing, cost and a timeline), `#exp-<id>` (one experiment in full) and `#terms`.
- **Source of truth:** [findings/registry.json](findings/registry.json). Edit that, not the HTML and not this file.
- **Build:** `~/.local/bin/uv run --project envs/judge python scripts/docs/build_dashboard.py` validates the registry and writes the page. Figures are SVGs in [findings/figs/](findings/figs/), drawn from derived numbers by `scripts/docs/make_figures.py`.

The earlier markdown version of this document (last state: commit `afe77a8`) was moved into the registry entry by entry on 2026-11-03.
