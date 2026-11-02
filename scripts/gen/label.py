"""Re-label a run set and rewrite its audit file and pilot summary (gen env; no GPU).

uv run --project envs/gen python scripts/gen/label.py --run-set m1_pilot
uv run --project envs/gen python scripts/gen/label.py --run-set m1_pilot --audit-per-group 10
uv run --project envs/gen python scripts/gen/label.py --run-set m1_pilot --audit-all
"""

import argparse
import json

from jmr import config
from jmr.analysis.pilot import analyse, render_markdown
from jmr.runset import label_run_set, read_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-set", required=True)
    parser.add_argument("--audit-per-group", type=int, default=5,
                        help="audit.md shows up to this many random runs per label × condition")
    parser.add_argument("--audit-all", action="store_true", help="audit.md shows every run")
    parser.add_argument("--audit-seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = config.run_set_dir(args.run_set)
    rows = label_run_set(args.run_set, audit_per_group=None if args.audit_all else args.audit_per_group,
                         audit_seed=args.audit_seed)
    meta_path = out_dir / "meta.json"
    pool = json.loads(meta_path.read_text()).get("pool_size") if meta_path.exists() else None
    summary = analyse(rows, read_items(out_dir), pool_size=pool)
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "pilot_summary.md").write_text(render_markdown(summary, args.run_set))
    print((out_dir / "pilot_summary.md").read_text())


if __name__ == "__main__":
    main()
