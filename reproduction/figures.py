"""
Figures for Section 5 -> paper/figures/

    e2_budget_sweep.pdf      (copied from the E2 run)
    robustness_violation.pdf (copied from the robustness run)
    real_cost.pdf            (written directly by experiments/cloud_cost.py)
    e1_latency_cdf.pdf       per-query latency of every strategy relative to Oracle
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduction.common import FIGURES, PRETTY, phase5, phase6, read_csv, tag_arg  # noqa: E402

CDF = ("mocap", "greedy_fastest", "greedy_cheapest", "random", "spark_default_cbo_off_aqe_on")


def copy(src: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        s = src.with_suffix(f".{ext}")
        if s.exists():
            FIGURES.mkdir(parents=True, exist_ok=True)
            shutil.copy(s, FIGURES / f"{name}.{ext}")
            print(f"wrote paper/figures/{name}.{ext}")


def latency_cdf(tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in read_csv(phase5(tag) / "e1_outcomes.csv") if r.get("oracle_regret")]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for s in CDF:
        vals = sorted(1.0 + float(r["oracle_regret"]) for r in rows if r["strategy"] == s)
        if not vals:
            continue
        ys = [(i + 1) / len(vals) for i in range(len(vals))]
        ax.step(vals, ys, where="post", label=PRETTY.get(s, s))
    ax.set_xscale("log")
    ax.set_xlabel("latency / Oracle latency")
    ax.set_ylabel("fraction of queries")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"e1_latency_cdf.{ext}", dpi=200)
    plt.close(fig)
    print("wrote paper/figures/e1_latency_cdf.pdf")


def main(argv=None) -> None:
    tag = tag_arg(argv)
    latency_cdf(tag)
    copy(phase5(tag) / "e2_budget_sweep.pdf", "e2_budget_sweep")
    copy(phase6(tag) / "robustness_violation.pdf", "robustness_violation")
    report = phase6("cloud_cost") / "cloud_cost_report.json"
    if report.exists():
        print("cloud cost:", json.dumps(json.loads(report.read_text()).get("telemetry_vs_bill_per_batch")))


if __name__ == "__main__":
    main()
