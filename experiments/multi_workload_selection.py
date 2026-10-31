"""
Multi-workload comparison of MOCAP selection policies.

Policies:
    - minimum predicted cost
    - minimum predicted latency
    - MOCAP Pareto + Lagrangian

All policies use the same candidate plans and calibrated predictions.

Workloads:
    Q1: asymmetric 80 / 800 / 8000
    Q2: reverse asymmetry 8000 / 800 / 80
    Q3: balanced 1000 / 2000 / 3000

No candidate query is executed.
"""

from __future__ import annotations

import csv
import os

import numpy as np
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
    "results/multi_workload_selection.csv"
)


WORKLOADS = {
    "Q1_asymmetric": (80, 800, 8000),
    "Q2_reverse": (8000, 800, 80),
    "Q3_balanced": (1000, 2000, 3000),
}


def make_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPMultiWorkloadSelection")
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
    workload_name: str,
    sizes: tuple[int, int, int],
) -> QueryRequest:

    n_size, c_size, o_size = sizes

    n = spark.range(0, n_size).toDF("n_id")
    c = spark.range(0, c_size).toDF("c_id")
    o = spark.range(0, o_size).toDF("o_id")

    n.createOrReplaceTempView(
        f"{workload_name}_n"
    )

    c.createOrReplaceTempView(
        f"{workload_name}_c"
    )

    o.createOrReplaceTempView(
        f"{workload_name}_o"
    )

    return QueryRequest(
        query_id=workload_name,
        sql=f"""
            SELECT *
            FROM {workload_name}_n n
            JOIN {workload_name}_c c
                ON n.n_id = c.c_id
            JOIN {workload_name}_o o
                ON c.c_id = o.o_id
        """,
        budget=1.0,
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


def choose_min_cost(feasible):

    if not feasible:
        return None

    return min(
        feasible,
        key=lambda item: item[1].estimated_cost,
    )


def choose_min_latency(feasible):

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

    pareto = (
        ParetoSelector
        .get_pareto_optimal(feasible)
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


def get_selected_info(selection):

    if selection is None:
        return "", "", ""

    plan, metrics = selection

    return (
        plan.plan_id,
        metrics.estimated_cost,
        metrics.estimated_latency,
    )


def generate_budgets(metrics):

    costs = sorted(
        float(metric.estimated_cost)
        for metric in metrics.values()
    )

    minimum = costs[0]
    maximum = costs[-1]

    # Use actual candidate-cost thresholds rather than
    # arbitrary absolute budgets.
    percentiles = np.percentile(
        costs,
        [0, 25, 50, 75, 100],
    )

    # Add a slightly tighter-than-minimum budget so that
    # the no-feasible case is explicitly tested.
    epsilon = max(
        minimum * 0.02,
        1e-12,
    )

    budgets = [
        max(minimum - epsilon, 1e-12),
        float(percentiles[0]),
        float(percentiles[1]),
        float(percentiles[2]),
        float(percentiles[4]),
    ]

    # Remove duplicates while preserving order.
    unique = []

    for budget in budgets:
        if not any(
            abs(budget - existing)
            <= max(abs(budget), 1e-15) * 1e-10
            for existing in unique
        ):
            unique.append(budget)

    return unique


def main():

    spark = make_spark()

    try:

        model = CalibrationModel.load(
            MODEL_PATH
        )

        rows = []

        print()
        print("=" * 80)
        print("MULTI-WORKLOAD SELECTION POLICY EVALUATION")
        print("=" * 80)

        for workload_name, sizes in WORKLOADS.items():

            print()
            print("-" * 80)
            print(
                f"{workload_name}: "
                f"sizes={sizes}"
            )
            print("-" * 80)

            base_query = create_query(
                spark,
                workload_name,
                sizes,
            )

            candidates = generate_join_candidates(
                base_query,
                spark,
            )

            metrics = estimate_candidates(
                spark,
                candidates,
                model,
            )

            budgets = generate_budgets(
                metrics
            )

            print(
                f"Candidates: {len(candidates)}"
            )

            print(
                "Candidate cost range: "
                f"{min(m.estimated_cost for m in metrics.values()):.10f}"
                " -> "
                f"{max(m.estimated_cost for m in metrics.values()):.10f}"
            )

            for budget in budgets:

                query = QueryRequest(
                    query_id=workload_name,
                    sql=base_query.sql,
                    budget=budget,
                    deadline=100.0,
                    accuracy_tolerance=0.0,
                )

                constraint_filter = (
                    ConstraintsFilter(
                        budget=budget,
                        deadline=query.deadline,
                    )
                )

                feasible = (
                    constraint_filter
                    .filter_feasible_plans(
                        candidates,
                        metrics,
                    )
                )

                selections = {
                    "min_cost":
                        choose_min_cost(
                            feasible
                        ),

                    "min_latency":
                        choose_min_latency(
                            feasible
                        ),

                    "mocap":
                        choose_mocap(
                            feasible,
                            budget,
                        ),
                }

                print()
                print(
                    f"budget={budget:.10f} "
                    f"feasible={len(feasible)}"
                )

                for policy, selection in (
                    selections.items()
                ):

                    plan_id, cost, latency = (
                        get_selected_info(
                            selection
                        )
                    )

                    print(
                        f"  {policy:12s} -> "
                        f"{plan_id or 'NONE'}"
                    )

                    rows.append(
                        {
                            "workload": workload_name,
                            "size_n": sizes[0],
                            "size_c": sizes[1],
                            "size_o": sizes[2],
                            "candidate_count":
                                len(candidates),
                            "budget": budget,
                            "feasible_count":
                                len(feasible),
                            "policy": policy,
                            "selected_plan":
                                plan_id,
                            "predicted_cost":
                                cost,
                            "predicted_latency":
                                latency,
                        }
                    )

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
                "workload",
                "size_n",
                "size_c",
                "size_o",
                "candidate_count",
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
