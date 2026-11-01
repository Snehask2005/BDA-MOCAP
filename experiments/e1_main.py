"""
E1 -- Main comparison table (owner: Anaswara, Phase 5).

MOCAP (conformal robust optimizer) vs Spark default (CBO/AQE on/off),
Greedy Cheapest, Greedy Fastest, Random, and Oracle on the same measured
candidate pools, at one budget multiplier and one delta.

Outputs (results/phase5/<tag>/):
    e1_outcomes.csv     one row per query x strategy
    e1_summary.csv      metric suite per strategy (median + 95% CI, P90/P99,
                        violation / feasibility rate, speedup, Oracle regret)
    e1_wilcoxon.csv     paired Wilcoxon tests MOCAP vs every baseline
                        (latency and cost, Holm-adjusted)
    e1_coverage.json    empirical leave-one-query-out conformal coverage
    e1_run_info.json    environment + arguments

Examples
--------
Laptop smoke test (synthetic data, small):
    python experiments/e1_main.py --scale 0.5 --table-counts 3 4 5 --repeats 3 --max-candidates 8

Cluster, official TPC-DS SF10:
    spark-submit experiments/e1_main.py --profile cluster --data tpcds_sf10 --repeats 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.cli import add_common_args, get_measurements, write_csv, write_run_info  # noqa: E402
from execution.harness import coverage_report, evaluate  # noqa: E402
from execution.stats import fmt, paired_tests, summarize_outcomes  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--budget-multiplier", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args(argv)

    env, measurements, out_dir = get_measurements(args, "e1_main")
    exp = env.experiment
    mult = args.budget_multiplier or float(exp.get("default_budget_multiplier", 1.5))
    delta = args.delta or float(exp.get("default_delta", 0.05))

    usable = [m for m in measurements if m.measured_pool()]
    skipped = [(m.query_id, m.error) for m in measurements if not m.measured_pool()]
    if len(usable) < 2:
        print("E1 needs at least 2 measured queries (leave-one-query-out calibration).")
        return 1

    outcomes = evaluate(usable, [mult], [delta], seed=env.seed)
    summary = summarize_outcomes(
        outcomes, ("strategy",), level=float(exp.get("confidence_level", 0.95)),
        n_resamples=int(exp.get("bootstrap_resamples", 2000)), seed=env.seed,
    )
    tests = paired_tests(outcomes, "mocap", "latency_s") + paired_tests(outcomes, "mocap", "cost")
    coverage = coverage_report(usable, delta)

    write_csv(out_dir / "e1_outcomes.csv", outcomes)
    write_csv(out_dir / "e1_summary.csv", summary)
    write_csv(out_dir / "e1_wilcoxon.csv", tests)
    (out_dir / "e1_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    write_run_info(out_dir / "e1_run_info.json", env, args, {
        "budget_multiplier": mult, "delta": delta,
        "queries_used": [m.query_id for m in usable], "queries_skipped": skipped,
    })

    print(f"\nE1  budget = {mult} x cheapest measured cost   delta = {delta}   "
          f"queries = {len(usable)}  (skipped {len(skipped)})")
    print(f"{'strategy':32} {'med lat(s)':>10} {'95% CI':>19} {'P99':>8} "
          f"{'viol%':>6} {'feas%':>6} {'regret':>7} {'speedup':>7}")
    for s in summary:
        ci = f"[{fmt(s['latency_ci_low'])}, {fmt(s['latency_ci_high'])}]"
        print(f"{s['strategy']:32} {fmt(s['median_latency_s']):>10} {ci:>19}"
              f" {fmt(s['p99_latency_s']):>8} {fmt(100 * s['budget_violation_rate'], 1):>6} "
              f"{fmt(100 * s['feasibility_rate'], 1):>6} {fmt(s['median_oracle_regret']):>7} "
              f"{fmt(s['geomean_speedup_vs_spark_default']):>7}")
    print("\nWilcoxon (MOCAP vs baseline, latency):")
    for t in tests:
        if t["metric"] == "latency_s":
            print(f"  vs {t['baseline']:32} wins={t['wins']:>3} losses={t['losses']:>3} "
                  f"ties={t['ties']:>3}  p={fmt(t['p_value'], 4)}  p_holm={fmt(t['p_value_holm'], 4)}")
    print(f"\nConformal cost coverage (target {coverage['target_coverage']:.2f}): "
          f"{fmt(coverage['conformal_coverage'])}  vs point prediction {fmt(coverage['point_coverage'])}")
    print(f"Results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
