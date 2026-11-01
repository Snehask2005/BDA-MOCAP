"""
1-click reproduction of every Section 5 table and figure (owner: Anaswara).

    # rebuild tables + figures from the cached raw measurements (no re-measuring)
    python reproduction/run_all.py --tag tpcds_sf10 --extra --data tpcds_sf10

    # re-measure everything from scratch first (needs Spark + data), incl. robustness
    python reproduction/run_all.py --tag tpcds_sf10 --remeasure --robustness \
        --extra --data tpcds_sf10 --profile cluster

Arguments after --extra go to every experiment script and must match the
ones used when measuring (data, scale, table counts, queries).

E1, E2 and E5 share one measurement cache, and the robustness suite has
its own (run it with --robustness), so after measuring once every later
command is fast. Existing robustness results are always turned into tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduction import (  # noqa: E402
    figures,
    table_e1_main,
    table_e2_budget,
    table_e5_case_study,
    table_robustness,
)
from reproduction.common import phase6, run  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--remeasure", action="store_true", help="re-run Spark measurements")
    parser.add_argument("--robustness", action="store_true",
                        help="also run experiments/robustness.py (long; measures 9 scenarios)")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="arguments forwarded to the experiment scripts (e.g. --data tpcds_sf10)")
    args = parser.parse_args(argv)

    forward = ["--tag", args.tag] + list(args.extra or [])
    if args.remeasure:
        forward.append("--no-reuse")
    run("experiments/e1_main.py", *forward)
    forward = [a for a in forward if a != "--no-reuse"]  # E2/E5 reuse E1's fresh measurements
    run("experiments/e2_budget.py", *forward)
    run("experiments/e5_case_study.py", *forward)
    if args.robustness:
        run("experiments/robustness.py", *(forward + (["--no-reuse"] if args.remeasure else [])))

    table_e1_main.main(["--tag", args.tag])
    table_e2_budget.main(["--tag", args.tag])
    table_e5_case_study.main(["--tag", args.tag])
    if (phase6(args.tag) / "robustness_summary.csv").exists():
        table_robustness.main(["--tag", args.tag])
    figures.main(["--tag", args.tag])
    print("\nAll Section 5 tables are in paper/tables/ and figures in paper/figures/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
