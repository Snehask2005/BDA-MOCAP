"""
Phase 6 -- Robustness stress suite (owner: Anaswara).

Question: does the conformal filter keep the budget-violation rate near
delta when the world differs from calibration, while the point-prediction
filter collapses? Every scenario is CALIBRATED on the clean world and
TESTED on the stressed one (that is the realistic deployment case).

Scenarios (pick with --scenarios):
    clean              fresh CBO statistics, uniform data (reference)
    missing_stats      no ANALYZE TABLE (CBO falls back to size heuristics)
    stale_stats        ANALYZE on the base data, then the data grows by
                       --growth x without refreshing statistics
    cardinality_low    prediction errors injected: each plan's predictions
                       multiplied by a plan-specific factor in [0.1, 1]
    cardinality_high   same with factors in [1, 10]
    zipf_skew          Zipf-skewed foreign keys (--zipf-s)
    data_shift         calibrate at the base scale, test at --growth x scale
                       (SF10 -> SF100 on the cluster via --shift-data)
    concurrent_2 / concurrent_4
                       measured while 2 / 4 background jobs load the cluster

Injected cardinality errors are reported in two variants: ``test_only``
(the estimator is wrong only at test time -- distribution shift) and
``calibrated`` (the same biased estimator is used for calibration, which
conformal prediction is designed to absorb).

Outputs (results/phase6/<tag>/):
    robustness_outcomes.csv, robustness_summary.csv, robustness_coverage.csv,
    robustness_violation.pdf/.png
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.cli import (  # noqa: E402
    add_common_args,
    apply_residual_mode,
    resolve_data_path,
    run_tag,
    workload,
    write_csv,
    write_run_info,
)
from execution.config import build_spark, load_env  # noqa: E402
from execution.data import (  # noqa: E402
    STAR_TABLES,
    analyze_tables,
    ensure_synthetic,
    make_stale_tables,
    register_parquet_dir,
)
from execution.harness import (  # noqa: E402
    background_load,
    calibrate,
    calibration_samples,
    coverage_report,
    evaluate,
    load_measurements,
    measure_workload,
)
from execution.run_plan import PlanRunner  # noqa: E402
from execution.stats import fmt, summarize_outcomes  # noqa: E402
from execution.telemetry import TelemetryLogger  # noqa: E402
from execution.workload import star_query_sql, SELECTIVITIES  # noqa: E402

ALL_SCENARIOS = ("clean", "missing_stats", "stale_stats", "cardinality_low", "cardinality_high",
                 "zipf_skew", "data_shift", "concurrent_2", "concurrent_4")
MEASURED = {"clean", "missing_stats", "stale_stats", "zipf_skew", "data_shift",
            "concurrent_2", "concurrent_4"}
STRATEGIES = ("mocap", "mocap_point", "greedy_fastest", "greedy_cheapest", "oracle")


def plan_factor(query_id: str, plan_id: str, low: float, high: float) -> float:
    """Deterministic plan-specific factor, log-uniform in [low, high]."""

    h = int(hashlib.sha256(f"{query_id}|{plan_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return math.exp(math.log(low) + h * (math.log(high) - math.log(low)))


def make_perturb(low: float, high: float) -> Callable:
    def perturb(qid, pid, cost, lat):
        f = plan_factor(qid, pid, low, high)
        return cost * f, lat * f
    return perturb


def prepare(spark, env, args, scenario: str) -> None:
    """Register data and statistics for one measured scenario."""

    base = resolve_data_path(env, args)
    synthetic = args.data == "synthetic"

    def synth(scale: float, zipf: float) -> str:
        root = str(Path(env.data_path("synthetic_root")) / f"scale{scale}_zipf{zipf}")
        return ensure_synthetic(spark, root, scale=scale, zipf_s=zipf, seed=env.seed)

    if synthetic:
        base = synth(args.scale, 0.0)

    if scenario in ("clean", "concurrent_2", "concurrent_4"):
        register_parquet_dir(spark, base, STAR_TABLES, mode="table")
        analyze_tables(spark, STAR_TABLES)
    elif scenario == "missing_stats":
        register_parquet_dir(spark, base, STAR_TABLES, mode="table")  # fresh tables: no stats
    elif scenario == "stale_stats":
        grown = synth(args.scale * args.growth, 0.0) if synthetic else args.shift_data
        if not grown:
            raise SystemExit("stale_stats on real data needs --shift-data (e.g. the SF100 path)")
        workdir = args.stale_workdir or str(Path(env.data_path("synthetic_root")) / "_stale_work")
        # Stats are computed on `base`, then the files are swapped for `grown`.
        report = make_stale_tables(spark, base, grown, workdir, STAR_TABLES)
        fact = report.get("store_sales", {})
        print(f"stale stats: store_sales catalog rows={fact.get('stats_rows')} "
              f"actual rows={fact.get('actual_rows')}")
        if fact.get("stats_rows") is None or fact.get("stats_rows") == fact.get("actual_rows"):
            raise SystemExit("stale_stats setup failed: statistics are not stale")
    elif scenario == "zipf_skew":
        skewed = synth(args.scale, args.zipf_s) if synthetic else args.skew_data
        if not skewed:
            raise SystemExit("zipf_skew on real data needs --skew-data")
        register_parquet_dir(spark, skewed, STAR_TABLES, mode="table")
        analyze_tables(spark, STAR_TABLES)
    elif scenario == "data_shift":
        shifted = synth(args.scale * args.growth, 0.0) if synthetic else args.shift_data
        if not shifted:
            raise SystemExit("data_shift on real data needs --shift-data (e.g. the SF100 path)")
        register_parquet_dir(spark, shifted, STAR_TABLES, mode="table")
        analyze_tables(spark, STAR_TABLES)
    else:
        raise ValueError(scenario)


def plot(summary, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    scen = sorted({s["scenario"] for s in summary})
    strategies = [s for s in ("mocap", "mocap_point", "greedy_fastest") if any(r["strategy"] == s for r in summary)]
    width = 0.8 / max(len(strategies), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(scen)), 3.4))
    for i, strategy in enumerate(strategies):
        vals = [next((100 * r["budget_violation_rate"] for r in summary
                      if r["scenario"] == sc and r["strategy"] == strategy), float("nan")) for sc in scen]
        ax.bar([x + i * width for x in range(len(scen))], vals, width, label=strategy)
    delta = summary[0]["delta"] if summary else 0.05
    ax.axhline(100 * delta, color="black", linestyle="--", linewidth=1, label=f"delta = {delta}")
    ax.set_xticks([x + width * (len(strategies) - 1) / 2 for x in range(len(scen))])
    ax.set_xticklabels(scen, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("budget violation rate (%)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"robustness_violation.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--scenarios", nargs="*", default=list(ALL_SCENARIOS), choices=ALL_SCENARIOS)
    parser.add_argument("--growth", type=float, default=4.0, help="data growth factor (2-10)")
    parser.add_argument("--zipf-s", type=float, default=1.2, help="Zipf exponent for zipf_skew")
    parser.add_argument("--shift-data", default=None, help="larger dataset (e.g. SF100) for real data")
    parser.add_argument("--skew-data", default=None, help="skewed dataset for real data")
    parser.add_argument("--stale-workdir", default=None,
                        help="writable working copy location for stale_stats (e.g. gs://bucket/mocap/stale)")
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--multipliers", nargs="*", type=float, default=None)
    args = parser.parse_args(argv)

    env = load_env(profile=args.profile)
    apply_residual_mode(env, args)
    exp = env.experiment
    delta = args.delta or float(exp.get("default_delta", 0.05))
    multipliers = args.multipliers or [float(x) for x in exp["budget_multipliers"]]
    tag = run_tag(args)
    out_dir = env.output_dir("phase6_dir") / tag
    queries = workload(args)

    measured = [s for s in args.scenarios if s in MEASURED]
    if any(s.startswith("cardinality") for s in args.scenarios) and "clean" not in measured:
        measured.insert(0, "clean")
    if "clean" not in measured:
        measured.insert(0, "clean")

    def cache_for(sc: str) -> Path:
        return out_dir / "measurements" / sc

    todo = [sc for sc in measured
            if args.no_reuse or not cache_for(sc).exists()
            or {p.stem for p in cache_for(sc).glob("*.json")} < {q.query_id for q in queries}]
    if todo:
        spark = build_spark(env, "robustness")
        try:
            logger = TelemetryLogger(env.output_dir("telemetry_dir") / f"robustness_{tag}.jsonl")
            for sc in todo:
                print(f"\n=== scenario {sc} ===")
                prepare(spark, env, args, sc)
                runner = PlanRunner(spark, env, logger, experiment="robustness", scenario=sc)
                kwargs = dict(repeats=args.repeats, warmup_runs=args.warmup_runs,
                              max_candidates=args.max_candidates, scenario=sc,
                              include_spark_defaults=False)
                if sc.startswith("concurrent_"):
                    n = int(sc.split("_")[1])
                    load_sql = star_query_sql(6, SELECTIVITIES["wide"])
                    with background_load(spark, load_sql, n):
                        measure_workload(spark, runner, queries, env, cache_dir=cache_for(sc),
                                         reuse=not args.no_reuse, **kwargs)
                else:
                    measure_workload(spark, runner, queries, env, cache_dir=cache_for(sc),
                                     reuse=not args.no_reuse, **kwargs)
        finally:
            spark.stop()

    wanted = {q.query_id for q in queries}
    data = {sc: [m for m in load_measurements(cache_for(sc)) if m.query_id in wanted and m.measured_pool()]
            for sc in measured}
    clean = data["clean"]
    if not clean:
        print("No clean measurements; cannot calibrate.")
        return 1

    outcomes: List[Dict] = []
    coverage: List[Dict] = []

    common = dict(strategies=STRATEGIES, include_spark_defaults=False)

    def record(label: str, rows) -> None:
        for r in rows:
            r["scenario"] = label
        outcomes.extend(rows)

    def stressed(label: str, tested, perturb=None) -> None:
        """Calibrate on the clean world, test on ``tested`` (+ optional perturbation)."""
        record(label, evaluate(tested, multipliers, [delta], seed=env.seed, perturb=perturb,
                               quantile_source=clean, **common))
        cq = calibrate(calibration_samples(clean), delta)[0]
        samples = calibration_samples(tested, perturb)
        if samples:
            coverage.append({
                "scenario": label, "delta": delta, "plans": len(samples),
                "cost_multiplier": cq.multiplier,
                "conformal_coverage": sum(s.actual_cost <= s.predicted_cost * cq.multiplier
                                          for s in samples) / len(samples),
                "point_coverage": sum(s.actual_cost <= s.predicted_cost for s in samples) / len(samples),
            })

    def loqo(label: str, perturb=None) -> None:
        """Leave-one-query-out on clean data (calibration sees the same estimator)."""
        record(label, evaluate(clean, multipliers, [delta], seed=env.seed, perturb=perturb, **common))
        cov = coverage_report(clean, delta, perturb)
        coverage.append({"scenario": label, "delta": delta, "plans": cov["plans"],
                         "cost_multiplier": None,
                         "conformal_coverage": cov["conformal_coverage"],
                         "point_coverage": cov["point_coverage"]})

    for sc in args.scenarios:
        if sc == "clean":
            loqo("clean")
        elif sc in ("cardinality_low", "cardinality_high"):
            lo, hi = (0.1, 1.0) if sc == "cardinality_low" else (1.0, 10.0)
            p = make_perturb(lo, hi)
            stressed(f"{sc}_test_only", clean, perturb=p)
            loqo(f"{sc}_calibrated", perturb=p)
        elif data.get(sc):
            stressed(sc, data[sc])
        else:
            print(f"scenario {sc}: no measurements, skipped")

    summary = summarize_outcomes(outcomes, ("scenario", "strategy"),
                                 n_resamples=int(exp.get("bootstrap_resamples", 2000)), seed=env.seed)
    for s in summary:
        s["delta"] = delta
    write_csv(out_dir / "robustness_outcomes.csv", outcomes)
    write_csv(out_dir / "robustness_summary.csv", summary)
    write_csv(out_dir / "robustness_coverage.csv", coverage)
    write_run_info(out_dir / "robustness_run_info.json", env, args, {"delta": delta, "multipliers": multipliers})
    plot(summary, out_dir)

    print(f"\nRobustness (delta = {delta}, multipliers = {multipliers})")
    print(f"{'scenario':32} {'strategy':16} {'viol%':>6} {'med lat':>8} {'regret':>7}")
    for s in summary:
        print(f"{s['scenario']:32} {s['strategy']:16} {fmt(100 * s['budget_violation_rate'], 1):>6} "
              f"{fmt(s['median_latency_s']):>8} {fmt(s['median_oracle_regret']):>7}")
    print(f"Results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
