"""Table: E1 main comparison -> paper/tables/e1_main.tex"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduction.common import (  # noqa: E402
    PRETTY,
    num,
    phase5,
    read_csv,
    sort_strategies,
    tag_arg,
    write_table,
)


def main(argv=None) -> None:
    tag = tag_arg(argv)
    rows = sort_strategies(read_csv(phase5(tag) / "e1_summary.csv"))
    tests = {(t["baseline"], t["metric"]): t for t in read_csv(phase5(tag) / "e1_wilcoxon.csv")}
    body = []
    for r in rows:
        t = tests.get((r["strategy"], "latency_s"))
        p = "--" if r["strategy"] == "mocap" or t is None else num(t["p_value_holm"], 3)
        body.append([
            PRETTY.get(r["strategy"], r["strategy"]),
            f"{num(r['median_latency_s'])} [{num(r['latency_ci_low'])}, {num(r['latency_ci_high'])}]",
            num(r["p99_latency_s"]),
            num(r["budget_violation_rate"], 1, 100),
            num(r["median_oracle_regret"], 3),
            num(r["geomean_speedup_vs_spark_default"], 2),
            p,
        ])
    write_table(
        "e1_main",
        "E1: plan selection quality on the same measured candidate pools. Median latency with "
        "95\\% bootstrap CI, P99 over all runs, budget-violation rate, median Oracle regret, "
        "geometric-mean speedup over Spark's default configuration, and Holm-adjusted Wilcoxon "
        "p-value of MOCAP vs.\\ each strategy.",
        "tab:e1_main",
        ["Strategy", "Median lat. (s) [95\\% CI]", "P99 (s)", "Viol. (\\%)", "Regret", "Speedup",
         "$p_{\\text{Holm}}$"],
        body, "lrrrrrr",
    )


if __name__ == "__main__":
    main()
