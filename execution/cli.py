"""Shared command-line setup for the experiment scripts (owner: Anaswara)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from execution.config import Env, build_spark, load_env, runtime_description
from execution.data import (
    STAR_TABLES,
    analyze_tables,
    ensure_synthetic,
    register_parquet_dir,
)
from execution.harness import json_safe
from execution.workload import WorkloadQuery, load_sql_dir, select_queries, star_join_workload


def add_common_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("environment")
    g.add_argument("--profile", default=None, help="configs/spark_env.yaml profile (local|cluster)")
    g.add_argument("--data", default="synthetic",
                   help="synthetic | tpcds_sf10 | tpcds_sf100 | <path to parquet dir>")
    g.add_argument("--scale", type=float, default=1.0, help="synthetic data scale")
    g.add_argument("--zipf", type=float, default=0.0, help="synthetic Zipf skew exponent")
    g.add_argument("--no-analyze", action="store_true",
                   help="skip ANALYZE TABLE (CBO statistics)")
    g = parser.add_argument_group("workload")
    g.add_argument("--queries", nargs="*", default=None,
                   help="query ids or families to run (default: all)")
    g.add_argument("--sql-dir", default=None, help="use *.sql files from this directory instead")
    g.add_argument("--table-counts", nargs="*", type=int, default=None,
                   help="star-join table counts (default 3..10)")
    g = parser.add_argument_group("protocol")
    g.add_argument("--repeats", type=int, default=None, help="measured runs per plan")
    g.add_argument("--warmup-runs", type=int, default=None)
    g.add_argument("--max-candidates", type=int, default=None)
    g.add_argument("--no-reuse", action="store_true", help="re-measure even if cached")
    g.add_argument("--tag", default=None, help="name of this run (output sub-folder)")
    g.add_argument("--residual", choices=["signed", "absolute"], default=None,
                   help="conformal score (default from configs/spark_env.yaml)")


def resolve_data_path(env: Env, args) -> str:
    if args.data == "synthetic":
        root = Path(env.data_path("synthetic_root")) / f"scale{args.scale}_zipf{args.zipf}"
        return str(root)
    if args.data in env.data:
        return env.data_path(args.data)
    return args.data


def setup(args, app: str):
    """Load env, start Spark, register (and if needed generate) the data."""

    env = load_env(profile=args.profile)
    spark = build_spark(env, app)
    data_path = resolve_data_path(env, args)
    if args.data == "synthetic":
        print(f"Synthetic TPC-DS-shaped data (NOT official TPC-DS): {data_path}")
        ensure_synthetic(spark, data_path, scale=args.scale, zipf_s=args.zipf, seed=env.seed)
    register_parquet_dir(spark, data_path, STAR_TABLES, mode="table")
    if not args.no_analyze:
        print("Computing CBO statistics (ANALYZE TABLE) ...")
        analyze_tables(spark, STAR_TABLES)
    return env, spark, data_path


def workload(args) -> List[WorkloadQuery]:
    if args.sql_dir:
        queries = load_sql_dir(args.sql_dir)
    else:
        counts = args.table_counts or list(range(3, 11))
        queries = star_join_workload(table_counts=counts)
    return select_queries(queries, args.queries)


def protocol_kwargs(args) -> Dict[str, Any]:
    return {
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "max_candidates": args.max_candidates,
    }


def apply_residual_mode(env: Env, args) -> str:
    from execution.harness import set_residual_mode

    mode = getattr(args, "residual", None) or env.experiment.get("conformal_residual", "signed")
    set_residual_mode(mode)
    env.experiment["conformal_residual"] = mode
    return mode


def run_tag(args) -> str:
    if args.tag:
        return args.tag
    if args.data == "synthetic":
        return f"synthetic_scale{args.scale}_zipf{args.zipf}"
    return Path(str(args.data)).name


def get_measurements(args, experiment: str, scenario: str = "baseline"):
    """
    Return (env, measurements, out_dir). Measurements are cached under
    results/phase5/<tag>/measurements and shared by E1, E2 and E5, so Spark
    is only started when something still needs to be measured.
    """

    from execution.harness import load_measurements, measure_workload
    from execution.run_plan import PlanRunner
    from execution.telemetry import TelemetryLogger

    env = load_env(profile=args.profile)
    apply_residual_mode(env, args)
    tag = run_tag(args)
    out_dir = env.output_dir("phase5_dir") / tag
    cache = out_dir / "measurements"
    queries = workload(args)
    cached = {p.stem for p in cache.glob("*.json")} if cache.exists() else set()
    missing = [q for q in queries if q.query_id not in cached]

    if missing or args.no_reuse:
        env, spark, _ = setup(args, experiment)
        try:
            log_path = env.output_dir("telemetry_dir") / f"{tag}.jsonl"
            runner = PlanRunner(spark, env, TelemetryLogger(log_path),
                                experiment=experiment, scenario=scenario)
            measure_workload(spark, runner, queries, env, cache_dir=cache,
                             reuse=not args.no_reuse, scenario=scenario, **protocol_kwargs(args))
        finally:
            spark.stop()

    wanted = {q.query_id for q in queries}
    measurements = [m for m in load_measurements(cache) if m.query_id in wanted]
    return env, measurements, out_dir


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json_safe(r) for r in rows]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_run_info(path: Path, env: Env, args, extra: Optional[Dict[str, Any]] = None) -> None:
    info = {"runtime": runtime_description(env), "args": vars(args),
            "conformal_residual": env.experiment.get("conformal_residual")}
    if extra:
        info.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
