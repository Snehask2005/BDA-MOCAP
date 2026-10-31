"""
Controlled actual-execution validation for the MOCAP baseline experiment.

Compares the plans selected at budget 4e-5 by:
  - Spark Default
  - Greedy Budget Pruning
  - Full MOCAP (Pareto + Lagrangian)

Configuration:
  - local[2]
  - AQE disabled
  - automatic broadcast disabled
  - 1 warm-up + 5 measured repetitions
  - randomized execution order
  - latency measured around the Spark action

Output:
  results/optimizer_actual_execution.csv
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------
# Spark/Python environment
# ---------------------------------------------------------------------

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession

from mocap.cost.estimator import observe_execution_telemetry
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


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

BUDGET = 0.000040
DEADLINE = 100.0

NUM_CORES = 2

WARMUP_RUNS = 1
MEASURED_REPETITIONS = 5

RANDOM_SEED = 42

MODEL_PATH = "results/end_to_end_calibration_model.json"

OUTPUT = Path(
    "results/optimizer_actual_execution.csv"
)


# ---------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------

def make_spark() -> SparkSession:

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName(
            "MOCAP-Actual-Baseline-Validation"
        )
        .config(
            "spark.ui.enabled",
            "false"
        )
        .config(
            "spark.sql.adaptive.enabled",
            "false"
        )
        .config(
            "spark.sql.autoBroadcastJoinThreshold",
            "-1"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(
        "ERROR"
    )

    return spark


# ---------------------------------------------------------------------
# Create the same baseline workload
# ---------------------------------------------------------------------

def create_query(
    spark: SparkSession,
) -> QueryRequest:

    n = (
        spark.range(0, 80)
        .toDF("n_id")
    )

    c = (
        spark.range(0, 800)
        .toDF("c_id")
    )

    o = (
        spark.range(0, 8000)
        .toDF("o_id")
    )

    n.createOrReplaceTempView(
        "baseline_n"
    )

    c.createOrReplaceTempView(
        "baseline_c"
    )

    o.createOrReplaceTempView(
        "baseline_o"
    )

    return QueryRequest(
        query_id="BASELINE_Q1",

        sql="""
            SELECT *
            FROM baseline_n n
            JOIN baseline_c c
                ON n.n_id = c.c_id
            JOIN baseline_o o
                ON c.c_id = o.o_id
        """,

        budget=BUDGET,

        deadline=DEADLINE,

        accuracy_tolerance=0.0,
    )


# ---------------------------------------------------------------------
# Candidate-specific prediction
# ---------------------------------------------------------------------

def estimate_candidates(
    spark,
    candidates,
    model,
):

    metrics = {}

    for candidate in candidates:

        dataframe = spark.sql(
            candidate.sql
        )

        statistics = (
            extract_plan_statistics(
                dataframe
            )
        )

        predicted = predict_candidates(
            [candidate],

            statistics=statistics,

            calibration_model=model,
        )

        metrics.update(
            predicted
        )

    return metrics


# ---------------------------------------------------------------------
# Reproduce the three policies
# ---------------------------------------------------------------------

def select_policies(
    query_id,
    candidates,
    metrics,
):

    # -------------------------------------------------------------
    # Spark default baseline
    # -------------------------------------------------------------

    spark_default = (
        SparkDefaultBaseline()
        .select(
            query_id=query_id,
            candidates=candidates,
            metrics=metrics,
            budget=BUDGET,
        )
    )

    # -------------------------------------------------------------
    # Greedy budget pruning
    # -------------------------------------------------------------

    greedy = (
        GreedyBudgetPruningBaseline(
            budget=BUDGET,
            deadline=DEADLINE,
        )
        .select(
            query_id=query_id,
            candidates=candidates,
            metrics=metrics,
        )
    )

    # -------------------------------------------------------------
    # Full MOCAP
    # -------------------------------------------------------------

    feasible = (
        ConstraintsFilter(
            budget=BUDGET,
            deadline=DEADLINE,
        )
        .filter_feasible_plans(
            candidates,
            metrics,
        )
    )

    pareto = (
        ParetoSelector.static_pareto(
            feasible
        )
    )

    ranked = (
        LagrangianRanker(
            alpha=0.5,
            beta=0.5,
        )
        .rank_candidates(
            pareto,
            BUDGET,
        )
    )

    if ranked:
        mocap_plan_id = (
            ranked[0][0].plan_id
        )
    else:
        mocap_plan_id = None

    # -------------------------------------------------------------
    # Selected plan IDs
    # -------------------------------------------------------------

    selected_ids = {

        "spark_default":
            spark_default.selected_plan_id,

        "greedy_budget":
            greedy.selected_plan_id,

        "mocap":
            mocap_plan_id,
    }

    # -------------------------------------------------------------
    # Map IDs → candidates
    # -------------------------------------------------------------

    candidate_by_id = {
        candidate.plan_id: candidate
        for candidate in candidates
    }

    selected = {

        policy: candidate_by_id[plan_id]

        for policy, plan_id
        in selected_ids.items()

        if plan_id is not None
    }

    return (
        selected,
        selected_ids,
        len(feasible),
        len(pareto),
    )


# ---------------------------------------------------------------------
# Execute one candidate once
# ---------------------------------------------------------------------

def execute_once(
    spark,
    candidate,
):

    dataframe = spark.sql(
        candidate.sql
    )

    start = time.perf_counter()

    row_count = dataframe.count()

    elapsed = (
        time.perf_counter()
        - start
    )

    telemetry = (
        observe_execution_telemetry(
            candidate=candidate,

            dataframe=dataframe,

            elapsed=elapsed,

            num_cores=NUM_CORES,
        )
    )

    return {

        "actual_latency_seconds":
            elapsed,

        "actual_cost":
            telemetry.actual_cost,

        "cpu_seconds":
            telemetry.cpu_seconds,

        "bytes_scanned":
            telemetry.bytes_scanned,

        "bytes_shuffled":
            telemetry.bytes_shuffled,

        "result_row_count":
            row_count,
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    spark = make_spark()

    try:

        # ---------------------------------------------------------
        # Load calibration model
        # ---------------------------------------------------------

        model = None

        if os.path.exists(
            MODEL_PATH
        ):

            model = (
                CalibrationModel.load(
                    MODEL_PATH
                )
            )

        # ---------------------------------------------------------
        # Create workload
        # ---------------------------------------------------------

        query = create_query(
            spark
        )

        # ---------------------------------------------------------
        # Generate candidates
        # ---------------------------------------------------------

        candidates = (
            generate_join_candidates(
                query,
                spark
            )
        )

        # ---------------------------------------------------------
        # Predict every candidate
        # ---------------------------------------------------------

        metrics = (
            estimate_candidates(
                spark,
                candidates,
                model,
            )
        )

        # ---------------------------------------------------------
        # Select policies
        # ---------------------------------------------------------

        (
            selected,
            selected_ids,
            feasible_count,
            pareto_count,
        ) = select_policies(
            query.query_id,
            candidates,
            metrics,
        )

        # ---------------------------------------------------------
        # Print experiment summary
        # ---------------------------------------------------------

        print()
        print("=" * 90)
        print(
            "MOCAP — ACTUAL EXECUTION VALIDATION"
        )
        print("=" * 90)

        print(
            f"Budget:       {BUDGET}"
        )

        print(
            f"Deadline:     {DEADLINE}s"
        )

        print(
            f"Candidates:   {len(candidates)}"
        )

        print(
            f"Feasible:     {feasible_count}"
        )

        print(
            f"Pareto:       {pareto_count}"
        )

        # ---------------------------------------------------------
        # Selected plans
        # ---------------------------------------------------------

        print()
        print("Selected plans:")

        for policy, plan_id in (
            selected_ids.items()
        ):

            if plan_id is None:

                print(
                    f"  {policy:<18} -> NONE"
                )

                continue

            candidate = next(
                c
                for c in candidates
                if c.plan_id == plan_id
            )

            metric = metrics[
                plan_id
            ]

            print(
                f"  {policy:<18} -> "
                f"{plan_id} | "
                f"{candidate.strategy} | "
                f"pred_cost="
                f"{metric.estimated_cost:.10f} | "
                f"pred_latency="
                f"{metric.estimated_latency:.6f}s"
            )

        # ---------------------------------------------------------
        # Remove duplicate selected plans
        #
        # If Greedy and MOCAP choose the same plan, we execute
        # that physical plan only once.
        # ---------------------------------------------------------

        unique_candidates = {
            candidate.plan_id: candidate
            for candidate in selected.values()
        }

        print()

        print(
            "Unique selected plans to execute: "
            f"{len(unique_candidates)}"
        )

        # ---------------------------------------------------------
        # Warm-up
        # ---------------------------------------------------------

        print()
        print("Warm-up:")

        for (
            plan_id,
            candidate
        ) in unique_candidates.items():

            result = execute_once(
                spark,
                candidate
            )

            print(
                f"  {plan_id}: "
                f"{result['actual_latency_seconds']:.6f}s"
            )

        # ---------------------------------------------------------
        # Measured repetitions
        # ---------------------------------------------------------

        rng = random.Random(
            RANDOM_SEED
        )

        plan_ids = list(
            unique_candidates
        )

        rows = []

        for repetition in range(
            1,
            MEASURED_REPETITIONS + 1
        ):

            order = plan_ids.copy()

            rng.shuffle(
                order
            )

            print()

            print(
                f"Measured repetition "
                f"{repetition}/"
                f"{MEASURED_REPETITIONS}"
            )

            for plan_id in order:

                candidate = (
                    unique_candidates[
                        plan_id
                    ]
                )

                result = execute_once(
                    spark,
                    candidate
                )

                metric = metrics[
                    plan_id
                ]

                policies = [

                    policy

                    for policy,
                    selected_candidate
                    in selected.items()

                    if (
                        selected_candidate.plan_id
                        == plan_id
                    )
                ]

                print(
                    f"  {plan_id:<22} "
                    f"actual="
                    f"{result['actual_latency_seconds']:.6f}s "
                    f"cost="
                    f"{result['actual_cost']:.10f} "
                    f"policies="
                    f"{','.join(policies)}"
                )

                # -------------------------------------------------
                # Save one row per policy/repetition
                # -------------------------------------------------

                for policy in policies:

                    rows.append({

                        "query_id":
                            query.query_id,

                        "budget":
                            BUDGET,

                        "deadline":
                            DEADLINE,

                        "policy":
                            policy,

                        "plan_id":
                            plan_id,

                        "strategy":
                            candidate.strategy,

                        "predicted_cost":
                            metric.estimated_cost,

                        "predicted_latency":
                            metric.estimated_latency,

                        "actual_latency_seconds":
                            result[
                                "actual_latency_seconds"
                            ],

                        "actual_cost":
                            result[
                                "actual_cost"
                            ],

                        "cpu_seconds":
                            result[
                                "cpu_seconds"
                            ],

                        "bytes_scanned":
                            result[
                                "bytes_scanned"
                            ],

                        "bytes_shuffled":
                            result[
                                "bytes_shuffled"
                            ],

                        "result_row_count":
                            result[
                                "result_row_count"
                            ],

                        "repetition":
                            repetition,

                        "warmup_runs":
                            WARMUP_RUNS,
                    })

        # ---------------------------------------------------------
        # Save CSV
        # ---------------------------------------------------------

        OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with OUTPUT.open(
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

        print()
        print("=" * 90)

        print(
            f"[DONE] Saved -> {OUTPUT}"
        )

        print("=" * 90)

    finally:

        spark.stop()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    main()
