"""
Compare MOCAP selection against simple feasible-plan baselines.

Policies:
    1. Minimum predicted cost
    2. Minimum predicted latency
    3. MOCAP: hard constraints -> Pareto -> Lagrangian

All policies operate on exactly the same generated candidates and
candidate-specific predicted metrics.

No candidate query is executed.
"""

from __future__ import annotations

import csv
import os

from pyspark.sql import SparkSession

from mocap.cost.learned import CalibrationModel
from mocap.cost.statistics import extract_plan_statistics
from mocap.cost.predictor import predict_candidates
from mocap.plans.generator import generate_join_candidates
from mocap.optimizer.constraints import ConstraintsFilter
from mocap.optimizer.pareto import ParetoSelector
from mocap.optimizer.lagrangian import LagrangianRanker
from mocap.query.parser import QueryRequest


MODEL_PATH = (
    "results/end_to_end_calibration_model.json"
)

OUTPUT = (
    "results/selection_policy_comparison.csv"
)

BUDGETS = [
    0.000024,
    0.000030,
    0.000040,
    0.000060,
    0.000100,
    0.000140,
]


def make_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPSelectionPolicyComparison")
        .config(
            "spark.sql.adaptive.enabled",
            "false",
        )
        .config(
            "spark.sql.autoBroadcastJoinThreshold",
            "-1",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def create_query(
    spark: SparkSession,
    budget: float,
) -> QueryRequest:

    n = spark.range(0, 80).toDF("n_id")
    c = spark.range(0, 800).toDF("c_id")
    o = spark.range(0, 8000).toDF("o_id")

    n.createOrReplaceTempView("policy_n")
    c.createOrReplaceTempView("policy_c")
    o.createOrReplaceTempView("policy_o")

    return QueryRequest(
        query_id="SELECTION_POLICY_Q1",
        sql="""
            SELECT *
            FROM policy_n n
            JOIN policy_c c
                ON n.n_id = c.c_id
            JOIN policy_o o
                ON c.c_id = o.o_id
        """,
        budget=budget,
        deadline=100.0,
        accuracy_tolerance=0.0,
    )


def estimate_candidates(
    spark,
    candidates,
    calibration_model,
):
    metrics = {}

    for candidate in candidates:

        dataframe = spark.sql(
            candidate.sql
        )

        statistics = extract_plan_statistics(
            dataframe
        )

        predicted = predict_candidates(
            [candidate],
            statistics=statistics,
            calibration_model=calibration_model,
        )

        metrics.update(predicted)

    return metrics


def choose_min_cost(
    feasible,
):
    if not feasible:
        return None

    return min(
        feasible,
        key=lambda item: item[1].estimated_cost,
    )


def choose_min_latency(
    feasible,
):
    if not feasible:
        return None

    return min(
        feasible,
        key=lambda item: item[1].estimated_latency,
    )


def choose_mocap(
    feasible,
    budget,
):
    if not feasible:
        return None

    pareto = ParetoSelector.get_pareto_optimal(
        feasible
    )

    if not pareto:
        return None

    ranker = LagrangianRanker(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        delta=0.0,
        lambda_mult=0.0,
    )

    ranked = ranker.rank_candidates(
        pareto,
        budget,
    )

    if not ranked:
        return None

    return ranked[0][:2]


def selected_info(selection):
    if selection is None:
        return "", "", ""

    plan, metrics = selection

    return (
        plan.plan_id,
        metrics.estimated_cost,
        metrics.estimated_latency,
    )


def main():
    spark = make_spark()

    try:
        model = CalibrationModel.load(
            MODEL_PATH
        )

        # ---------------------------------------------------------
        # Generate ONE candidate set for the workload.
        # ---------------------------------------------------------

        base_query = create_query(
            spark,
            budget=BUDGETS[-1],
        )

        candidates = generate_join_candidates(
            base_query,
            spark,
        )

        print()
        print("=" * 80)
        print("SELECTION POLICY COMPARISON")
        print("=" * 80)

        print(
            f"Candidate count: {len(candidates)}"
        )

        rows = []

        for budget in BUDGETS:

            query = create_query(
                spark,
                budget=budget,
            )

            # Same candidates, same prediction model.
            metrics = estimate_candidates(
                spark,
                candidates,
                model,
            )

            constraint_filter = ConstraintsFilter(
                budget=budget,
                deadline=query.deadline,
            )

            feasible = (
                constraint_filter
                .filter_feasible_plans(
                    candidates,
                    metrics,
                )
            )

            selections = {
                "min_cost": choose_min_cost(
                    feasible
                ),
                "min_latency": choose_min_latency(
                    feasible
                ),
                "mocap": choose_mocap(
                    feasible,
                    budget,
                ),
            }

            print()
            print(
                f"Budget: {budget:.8f}"
            )
            print(
                f"Feasible candidates: "
                f"{len(feasible)}"
            )

            for policy, selection in selections.items():

                plan_id, cost, latency = (
                    selected_info(selection)
                )

                print(
                    f"  {policy:12s} -> "
                    f"{plan_id or 'NONE'}"
                )

                rows.append(
                    {
                        "query_id": query.query_id,
                        "budget": budget,
                        "feasible_count": len(feasible),
                        "policy": policy,
                        "selected_plan": plan_id,
                        "predicted_cost": cost,
                        "predicted_latency": latency,
                    }
                )

        # ---------------------------------------------------------
        # Save
        # ---------------------------------------------------------

        os.makedirs(
            os.path.dirname(OUTPUT),
            exist_ok=True,
        )

        with open(
            OUTPUT,
            "w",
            newline="",
        ) as file:

            fieldnames = [
                "query_id",
                "budget",
                "feasible_count",
                "policy",
                "selected_plan",
                "predicted_cost",
                "predicted_latency",
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        print()
        print(
            f"Saved results to {OUTPUT}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
