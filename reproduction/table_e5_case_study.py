"""Table: E5 per-family MOCAP vs Greedy -> paper/tables/e5_case_study.tex"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduction.common import PRETTY, latex_escape, num, phase5, read_csv, tag_arg, write_table  # noqa: E402


def main(argv=None) -> None:
    tag = tag_arg(argv)
    rows = read_csv(phase5(tag) / "e5_case_study.csv")
    mults = sorted({float(r["budget_multiplier"]) for r in rows})
    m = 1.5 if 1.5 in mults else mults[len(mults) // 2]
    body = []
    for r in rows:
        if abs(float(r["budget_multiplier"]) - m) > 1e-12:
            continue
        body.append([
            latex_escape(r["family"]),
            PRETTY.get(r["greedy"], r["greedy"]),
            r["n_queries"],
            f"{r['wins']}/{r['losses']}/{r['ties']}",
            num(r["p_value"], 4),
            f"{r['mocap_violations']}/{r['greedy_violations']}",
        ])
    write_table(
        "e5_case_study",
        f"E5: MOCAP vs.\\ Greedy per query family at $m={m:g}$: wins/losses/ties on median latency, "
        "two-sided Wilcoxon p-value, and budget violations (MOCAP/Greedy).",
        "tab:e5_case_study",
        ["Family", "vs.", "$n$", "W/L/T", "$p$", "Viol."],
        body, "llrrrr",
    )


if __name__ == "__main__":
    main()
