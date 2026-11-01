"""Table: E2 budget sweep at the default delta -> paper/tables/e2_budget.tex"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduction.common import PRETTY, num, phase5, read_csv, tag_arg, write_table  # noqa: E402

SHOW = ("mocap", "mocap_point", "greedy_fastest", "greedy_cheapest", "oracle")


def main(argv=None) -> None:
    tag = tag_arg(argv)
    rows = read_csv(phase5(tag) / "e2_summary.csv")
    deltas = sorted({float(r["delta"]) for r in rows})
    delta = 0.05 if 0.05 in deltas else deltas[0]
    mults = sorted({float(r["budget_multiplier"]) for r in rows})
    index = {(r["strategy"], float(r["budget_multiplier"])): r for r in rows
             if abs(float(r["delta"]) - delta) < 1e-12}
    body = []
    for s in SHOW:
        cells = [PRETTY[s]]
        for m in mults:
            r = index.get((s, m))
            cells.append("--" if r is None else
                         f"{num(r['budget_violation_rate'], 1, 100)} / {num(r['median_latency_s'], 2)}")
        body.append(cells)
    write_table(
        "e2_budget",
        f"E2: budget-violation rate (\\%) / median latency (s) as the budget grows from 1.0$\\times$ "
        f"to 2.0$\\times$ the cheapest measured cost ($\\delta = {delta}$).",
        "tab:e2_budget",
        ["Strategy"] + [f"$m={m:g}$" for m in mults],
        body, "l" + "r" * len(mults),
    )


if __name__ == "__main__":
    main()
