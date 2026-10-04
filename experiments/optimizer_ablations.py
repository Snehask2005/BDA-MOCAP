"""
Ablation experiment for the MOCAP optimizer.

Roshini (Member 3) — Core MOCAP Optimizer

Runs every registered ablation variant against the same workload and
budget sweep to measure the contribution of each component.

Ablation variants:
    full_mocap       — baseline (all components enabled)
    no_calibration   — analytical predictor only
    no_pareto        — all feasible plans enter the ranker
    no_lagrangian    — min-cost selection instead of Lagrangian
    no_adaptive      — accuracy_tolerance forced to 0
    latency_only     — alpha=1, beta=0
    cost_only        — alpha=0, beta=1
    high_lambda      — violation penalty lambda=10

Outputs (saved to results/):
    optimizer_ablations.csv  — per-variant, per-budget row

Usage:
    python -m experiments.optimizer_ablations
"""

from __future__ import annotations

import csv
import os
import sys
import time

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession

from mocap.cost.learned import CalibrationModel
from mocap.cost.predictor import predict_candidates
from mocap.cost.statistics import extract_plan_statistics
from mocap.optimizer.ablations import ALL_ABLATIONS, AblationVariant
from mocap.optimizer.selector import MOCAPSelector
from mocap.plans.generator import generate_join_candidates
from mocap.query.parser import QueryRequest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "results/end_to_end_calibration_model.json"
OUTPUT_CSV = "results/optimizer_ablations.csv"

BUDGETS = [
    0.000022,
    0.000040,
    0.000100,
    1.0,
]

DEADLINE = 100.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPAblations")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def create_query(spark: SparkSession, budget: float) -> QueryRequest:
    n = spark.range(0, 80).toDF("n_id")
    c = spark.range(0, 800).toDF("c_id")
    o = spark.range(0, 8000).toDF("o_id")
    n.createOrReplaceTempView("ablation_n")
    c.createOrReplaceTempView("ablation_c")
    o.createOrReplaceTempView("ablation_o")
    return QueryRequest(
        query_id="ABLATION_Q1",
        sql="""
            SELECT *
            FROM ablation_n n
            JOIN ablation_c c ON n.n_id = c.c_id
            JOIN ablation_o o ON c.c_id = o.o_id
        """,
        budget=budget,
        deadline=DEADLINE,
        accuracy_tolerance=5.0,  # non-zero so NO_ADAPTIVE ablation has visible effect
    )


def estimate_candidates(spark, candidates, model):
    metrics = {}
    for candidate in candidates:
        df = spark.sql(candidate.sql)
        stats = extract_plan_statistics(df)
        predicted = predict_candidates(
            [candidate],
            statistics=stats,
            calibration_model=model,
        )
        metrics.update(predicted)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    spark = make_spark()

    try:
        model = None
        if os.path.exists(MODEL_PATH):
            try:
                model = CalibrationModel.load(MODEL_PATH)
            except Exception:
                pass

        base_query = create_query(spark, budget=1.0)
        candidates = generate_join_candidates(base_query, spark)

        # Use calibrated metrics for all ablations.
        # The no_calibration ablation passes model=None at selection time.
        metrics_calibrated = estimate_candidates(spark, candidates, model)
        metrics_analytical = estimate_candidates(spark, candidates, None)

        print()
        print("=" * 70)
        print("MOCAP — ABLATION EXPERIMENT")
        print("=" * 70)
        print(f"Candidates: {len(candidates)}")
        print(f"Ablations:  {len(ALL_ABLATIONS)}")

        rows = []

        for variant in AblationVariant:
            cfg = ALL_ABLATIONS[variant]

            # Choose metric set: no_calibration ablation uses analytical.
            use_metrics = (
                metrics_analytical
                if variant == AblationVariant.NO_CALIBRATION
                else metrics_calibrated
            )

            selector = MOCAPSelector(ablation=cfg)

            print()
            print(f"  [{variant.value}]")

            for budget in BUDGETS:
                query = create_query(spark, budget=budget)

                result = selector.select(
                    query_id=query.query_id,
                    candidates=candidates,
                    metrics=use_metrics,
                    budget=budget,
                    deadline=DEADLINE,
                    accuracy_tolerance=query.accuracy_tolerance,
                )

                status = "SELECTED" if result.selected_plan else "INFEASIBLE"
                print(
                    f"    budget={budget:.6f} | "
                    f"feasible={result.feasible_count} | "
                    f"pareto={result.pareto_count} | "
                    f"{status}"
                )

                rows.append({
                    "ablation_variant": variant.value,
                    "description": cfg.description[:80],
                    "use_calibration": cfg.use_calibration,
                    "use_pareto": cfg.use_pareto,
                    "use_lagrangian": cfg.use_lagrangian,
                    "alpha": cfg.alpha,
                    "beta": cfg.beta,
                    "lambda_mult": cfg.lambda_mult,
                    "budget": budget,
                    "candidate_count": result.candidate_count,
                    "feasible_count": result.feasible_count,
                    "pareto_count": result.pareto_count,
                    "selected_plan": (
                        result.selected_plan.plan_id if result.selected_plan else ""
                    ),
                    "selected_strategy": (
                        result.selected_plan.selected_strategy
                        if result.selected_plan
                        else ""
                    ),
                    "expected_cost": (
                        result.selected_plan.expected_cost
                        if result.selected_plan
                        else ""
                    ),
                    "expected_latency": (
                        result.selected_plan.expected_latency
                        if result.selected_plan
                        else ""
                    ),
                    "best_score": (
                        result.best_score if result.best_score is not None else ""
                    ),
                    "total_optimizer_time_s": result.total_time_s,
                    "pareto_dominated": result.pareto_dominated_count,
                    "failure_reason": result.failure_reason or "",
                })

        # ---------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print()
        print(f"[DONE] Saved -> {OUTPUT_CSV}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
