"""
E2 -- Budget sweep (owner: Anaswara, Phase 5).

Budgets are set RELATIVE to the Oracle (guardrail: never arbitrary
constants): B = m x cheapest measured cost of the query, for
m in {1.0, 1.2, 1.5, 2.0}, crossed with delta in {0.01, 0.05, 0.10, 0.20}.

Reuses E1's cached measurements (same --tag / data settings), so it costs no
extra Spark time when run after E1.

Outputs (results/phase5/<tag>/):
    e2_outcomes.csv   one row per query x strategy x m x delta
    e2_summary.csv    metric suite per (strategy, m, delta)
    e2_wilcoxon.csv   MOCAP vs baselines per (m, delta), Holm-adjusted
    e2_budget_sweep.pdf/.png   violation rate and latency vs m (if matplotlib)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.cli import add_common_args, get_measurements, write_csv, write_run_info  # noqa: E402
from execution.harness import evaluate  # noqa: E402
from execution.stats import fmt, paired_tests, summarize_outcomes  # noqa: E402

FOCUS = ("mocap", "mocap_point", "greedy_fastest", "greedy_cheapest", "random", "oracle",
         "spark_default_cbo_off_aqe_on")


def plot(summary, out_dir: Path, delta: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    rows = [s for s in summary if abs(s["delta"] - delta) < 1e-12 and s["strategy"] in FOCUS]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for strategy in FOCUS:
        pts = sorted((s["budget_multiplier"], s) for s in rows if s["strategy"] == strategy)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        axes[0].plot(xs, [100 * p[1]["budget_violation_rate"] for p in pts], marker="o", label=strategy)
        axes[1].plot(xs, [p[1]["median_latency_s"] for p in pts], marker="o", label=strategy)
    axes[0].set_xlabel("budget multiplier m (x Oracle cheapest cost)")
    axes[0].set_ylabel("budget violation rate (%)")
    axes[1].set_xlabel("budget multiplier m")
    axes[1].set_ylabel("median latency (s)")
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle(f"E2 budget sweep (delta = {delta})", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"e2_budget_sweep.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--multipliers", nargs="*", type=float, default=None)
    parser.add_argument("--deltas", nargs="*", type=float, default=None)
    args = parser.parse_args(argv)

    env, measurements, out_dir = get_measurements(args, "e2_budget")
    exp = env.experiment
    multipliers = args.multipliers or [float(x) for x in exp["budget_multipliers"]]
    deltas = args.deltas or [float(x) for x in exp["deltas"]]
    usable = [m for m in measurements if m.measured_pool()]
    if len(usable) < 2:
        print("E2 needs at least 2 measured queries.")
        return 1

    outcomes = evaluate(usable, multipliers, deltas, seed=env.seed)
    keys = ("strategy", "budget_multiplier", "delta")
    summary = summarize_outcomes(outcomes, keys, n_resamples=int(exp.get("bootstrap_resamples", 2000)),
                                 seed=env.seed)
    tests = paired_tests(outcomes, "mocap", "latency_s", group_keys=("budget_multiplier", "delta"))

    write_csv(out_dir / "e2_outcomes.csv", outcomes)
    write_csv(out_dir / "e2_summary.csv", summary)
    write_csv(out_dir / "e2_wilcoxon.csv", tests)
    write_run_info(out_dir / "e2_run_info.json", env, args,
                   {"multipliers": multipliers, "deltas": deltas})
    plot(summary, out_dir, float(exp.get("default_delta", 0.05)) if
         float(exp.get("default_delta", 0.05)) in deltas else deltas[0])

    print(f"\nE2 budget sweep over {len(usable)} queries")
    print(f"{'delta':>6} {'m':>4}  {'strategy':30} {'viol%':>6} {'feas%':>6} {'med lat':>8} {'regret':>7}")
    for s in summary:
        if s["strategy"] not in FOCUS:
            continue
        print(f"{s['delta']:>6} {s['budget_multiplier']:>4}  {s['strategy']:30} "
              f"{fmt(100 * s['budget_violation_rate'], 1):>6} {fmt(100 * s['feasibility_rate'], 1):>6} "
              f"{fmt(s['median_latency_s']):>8} {fmt(s['median_oracle_regret']):>7}")
    print(f"Results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
