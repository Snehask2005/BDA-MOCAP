"""
Baseline comparison experiment for the MOCAP optimizer.

Roshini (Member 3) — Core MOCAP Optimizer

Compares MOCAP multi-objective selection against two baselines:
    1. SparkDefaultBaseline    — minimum-cost, no budget awareness
    2. GreedyBudgetPruningBaseline — minimum-latency within budget

All policies use identical candidate plans and predicted metrics.
No candidate query is executed.

Outputs (saved to results/):
    optimizer_baselines.csv  — per-budget, per-policy row

Usage:
    python -m experiments.optimizer_baselines
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
from mocap.optimizer.baselines import (
    GreedyBudgetPruningBaseline,
    SparkDefaultBaseline,
)
from mocap.optimizer.constraints import ConstraintsFilter
from mocap.optimizer.lagrangian import LagrangianRanker
from mocap.optimizer.pareto import ParetoSelector
from mocap.plans.generator import generate_join_candidates
from mocap.query.parser import QueryRequest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "results/end_to_end_calibration_model.json"
OUTPUT_CSV = "results/optimizer_baselines.csv"

BUDGETS = [
    0.000020,
    0.000024,
    0.000030,
    0.000040,
    0.000060,
    0.000100,
    0.000140,
]

DEADLINE = 100.0


# ---------------------------------------------------------------------------
# Spark / query helpers
# ---------------------------------------------------------------------------

def make_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPBaselines")
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
    n.createOrReplaceTempView("baseline_n")
    c.createOrReplaceTempView("baseline_c")
    o.createOrReplaceTempView("baseline_o")
    return QueryRequest(
        query_id="BASELINE_Q1",
        sql="""
            SELECT *
            FROM baseline_n n
            JOIN baseline_c c ON n.n_id = c.c_id
            JOIN baseline_o o ON c.c_id = o.o_id
        """,
        budget=budget,
        deadline=DEADLINE,
        accuracy_tolerance=0.0,
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


def select_mocap(feasible, budget):
    """MOCAP: Pareto + Lagrangian."""
    if not feasible:
        return None, None, None
    pareto = ParetoSelector.static_pareto(feasible)
    if not pareto:
        return None, None, None
    ranker = LagrangianRanker(alpha=0.5, beta=0.5)
    ranked = ranker.rank_candidates(pareto, budget)
    if not ranked:
        return None, None, None
    best_plan, best_metrics, best_score = ranked[0]
    return best_plan.plan_id, best_metrics.estimated_cost, best_metrics.estimated_latency


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
        metrics = estimate_candidates(spark, candidates, model)

        print()
        print("=" * 70)
        print("MOCAP — BASELINE COMPARISON")
        print("=" * 70)
        print(f"Candidates: {len(candidates)}")

        spark_default = SparkDefaultBaseline()
        rows = []

        for budget in BUDGETS:
            # ---------------------------------------------------------------
            # Baseline 1: SparkDefault (no budget awareness)
            # ---------------------------------------------------------------
            r_spark = spark_default.select(
                query_id="BASELINE_Q1",
                candidates=candidates,
                metrics=metrics,
                budget=budget,
            )

            # ---------------------------------------------------------------
            # Baseline 2: GreedyBudgetPruning
            # ---------------------------------------------------------------
            greedy = GreedyBudgetPruningBaseline(budget=budget, deadline=DEADLINE)
            r_greedy = greedy.select(
                query_id="BASELINE_Q1",
                candidates=candidates,
                metrics=metrics,
            )

            # ---------------------------------------------------------------
            # MOCAP
            # ---------------------------------------------------------------
            cf = ConstraintsFilter(budget=budget, deadline=DEADLINE)
            feasible = cf.filter_feasible_plans(candidates, metrics)
            mocap_id, mocap_cost, mocap_lat = select_mocap(feasible, budget)

            print()
            print(f"Budget: {budget:.8f}  Feasible: {len(feasible)}")
            print(f"  spark_default   -> {r_spark.selected_plan_id or 'NONE'}")
            print(f"  greedy_budget   -> {r_greedy.selected_plan_id or 'NONE'}")
            print(f"  mocap           -> {mocap_id or 'NONE'}")

            for policy, plan_id, cost, latency in [
                ("spark_default",   r_spark.selected_plan_id,  r_spark.predicted_cost,  r_spark.predicted_latency),
                ("greedy_budget",   r_greedy.selected_plan_id, r_greedy.predicted_cost, r_greedy.predicted_latency),
                ("mocap",           mocap_id,                  mocap_cost,              mocap_lat),
            ]:
                rows.append({
                    "query_id": "BASELINE_Q1",
                    "budget": budget,
                    "feasible_count": len(feasible),
                    "policy": policy,
                    "selected_plan": plan_id or "",
                    "predicted_cost": cost if cost is not None else "",
                    "predicted_latency": latency if latency is not None else "",
                    "cost_within_budget": (
                        "YES" if (cost is not None and cost <= budget) else
                        ("NO" if cost is not None else "N/A")
                    ),
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
