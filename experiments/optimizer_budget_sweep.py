"""
Budget sweep experiment for the MOCAP optimizer.

Roshini (Member 3) — Core MOCAP Optimizer

Runs the same workload under strict / moderate / relaxed budgets and
an infeasible budget to characterize optimizer behaviour across the
full budget spectrum.

Outputs (saved to results/):
    optimizer_budget_sweep.csv  — per-budget row with counts and selection
    optimizer_budget_sweep.json — full SelectionResult dicts for debugging

Usage:
    python -m experiments.optimizer_budget_sweep
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession

from mocap.cost.learned import CalibrationModel
from mocap.cost.predictor import predict_candidates
from mocap.cost.statistics import extract_plan_statistics
from mocap.optimizer.selector import MOCAPSelector
from mocap.plans.generator import generate_join_candidates
from mocap.query.parser import QueryRequest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "results/end_to_end_calibration_model.json"
OUTPUT_CSV = "results/optimizer_budget_sweep.csv"
OUTPUT_JSON = "results/optimizer_budget_sweep.json"

# Budget categories: strict < moderate < relaxed + one infeasible.
BUDGET_LABELS = [
    ("infeasible", 0.0000001),
    ("strict",     0.000022),
    ("moderate",   0.000040),
    ("relaxed",    0.000100),
    ("generous",   1.0),
]

DEADLINE = 100.0   # seconds
ACCURACY_TOLERANCE = 0.0


# ---------------------------------------------------------------------------
# Spark setup
# ---------------------------------------------------------------------------

def make_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPOptimizerBudgetSweep")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---------------------------------------------------------------------------
# Query factory
# ---------------------------------------------------------------------------

def create_query(spark: SparkSession, budget: float) -> QueryRequest:
    n = spark.range(0, 80).toDF("n_id")
    c = spark.range(0, 800).toDF("c_id")
    o = spark.range(0, 8000).toDF("o_id")
    n.createOrReplaceTempView("sweep_n")
    c.createOrReplaceTempView("sweep_c")
    o.createOrReplaceTempView("sweep_o")
    return QueryRequest(
        query_id="BUDGET_SWEEP_Q1",
        sql="""
            SELECT *
            FROM sweep_n n
            JOIN sweep_c c ON n.n_id = c.c_id
            JOIN sweep_o o ON c.c_id = o.o_id
        """,
        budget=budget,
        deadline=DEADLINE,
        accuracy_tolerance=ACCURACY_TOLERANCE,
    )


# ---------------------------------------------------------------------------
# Candidate estimation
# ---------------------------------------------------------------------------

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
        # Load calibration model if available.
        model = None
        if os.path.exists(MODEL_PATH):
            try:
                model = CalibrationModel.load(MODEL_PATH)
                print(f"[INFO] Loaded calibration model from {MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Could not load calibration model: {e}")
        else:
            print("[INFO] No calibration model found — using analytical predictor.")

        # Generate candidates once (same for all budgets).
        base_query = create_query(spark, budget=1.0)
        t0 = time.perf_counter()
        candidates = generate_join_candidates(base_query, spark)
        candidate_gen_time = time.perf_counter() - t0

        print()
        print("=" * 70)
        print("MOCAP OPTIMIZER — BUDGET SWEEP")
        print("=" * 70)
        print(f"Candidates generated: {len(candidates)}")
        print(f"Candidate generation time: {candidate_gen_time:.4f} s")

        # Predict metrics once.
        t0 = time.perf_counter()
        metrics = estimate_candidates(spark, candidates, model)
        prediction_time = time.perf_counter() - t0
        print(f"Prediction time: {prediction_time:.4f} s")

        cost_values = sorted(m.estimated_cost for m in metrics.values())
        print(f"Predicted cost range: [{cost_values[0]:.8f}, {cost_values[-1]:.8f}]")

        rows = []
        json_records = []

        selector = MOCAPSelector(alpha=0.5, beta=0.5)

        for label, budget in BUDGET_LABELS:
            query = create_query(spark, budget=budget)

            result = selector.select(
                query_id=query.query_id,
                candidates=candidates,
                metrics=metrics,
                budget=budget,
                deadline=DEADLINE,
                accuracy_tolerance=ACCURACY_TOLERANCE,
            )

            print()
            print(f"  Budget [{label:10s}] = {budget:.8f}")
            print(f"    Feasible:     {result.feasible_count} / {result.candidate_count}")
            print(f"    Pareto:       {result.pareto_count}")
            print(
                f"    Selected:     "
                + (result.selected_plan.plan_id if result.selected_plan else "NONE")
            )
            if result.failure_reason:
                print(f"    Failure:      {result.failure_reason}")

            row = {
                "budget_label": label,
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
                "best_score": result.best_score if result.best_score is not None else "",
                "constraint_time_s": result.constraint_time_s,
                "pareto_time_s": result.pareto_time_s,
                "ranking_time_s": result.ranking_time_s,
                "total_optimizer_time_s": result.total_time_s,
                "pareto_dominated": result.pareto_dominated_count,
                "failure_reason": result.failure_reason or "",
            }
            rows.append(row)
            json_records.append(result.to_dict())

        # ---------------------------------------------------------------
        # Save results
        # ---------------------------------------------------------------
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        with open(OUTPUT_JSON, "w") as f:
            json.dump(json_records, f, indent=2, default=str)

        print()
        print(f"[DONE] CSV  -> {OUTPUT_CSV}")
        print(f"[DONE] JSON -> {OUTPUT_JSON}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
