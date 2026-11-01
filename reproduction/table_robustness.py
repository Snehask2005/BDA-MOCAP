"""Table: Phase 6 robustness -> paper/tables/robustness.tex"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduction.common import latex_escape, num, phase6, read_csv, tag_arg, write_table  # noqa: E402


def main(argv=None) -> None:
    tag = tag_arg(argv)
    summary = read_csv(phase6(tag) / "robustness_summary.csv")
    coverage = {r["scenario"]: r for r in read_csv(phase6(tag) / "robustness_coverage.csv")}
    viol = {(r["scenario"], r["strategy"]): r["budget_violation_rate"] for r in summary}
    scenarios = sorted({r["scenario"] for r in summary}, key=lambda s: (s != "clean", s))
    body = []
    for sc in scenarios:
        cov = coverage.get(sc, {})
        body.append([
            latex_escape(sc),
            num(cov.get("conformal_coverage"), 3),
            num(cov.get("point_coverage"), 3),
            num(viol.get((sc, "mocap")), 1, 100),
            num(viol.get((sc, "mocap_point")), 1, 100),
            num(viol.get((sc, "greedy_fastest")), 1, 100),
        ])
    delta = summary[0].get("delta", "0.05") if summary else "0.05"
    write_table(
        "robustness",
        f"Robustness ($\\delta = {delta}$): empirical cost-bound coverage of the conformal vs.\\ point "
        "prediction, and budget-violation rate (\\%) of MOCAP, MOCAP with a point filter, and Greedy "
        "Fastest. Every stressed scenario is calibrated on the clean world.",
        "tab:robustness",
        ["Scenario", "Cov. conf.", "Cov. point", "Viol. MOCAP", "Viol. point", "Viol. Greedy"],
        body, "lrrrrr",
    )


if __name__ == "__main__":
    main()
