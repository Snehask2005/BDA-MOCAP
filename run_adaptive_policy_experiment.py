"""
Execution-policy experiment for MOCAP (Strict vs Adaptive/Degraded).

For one query:
  1. Plan with MOCAP using a relaxed planning budget to obtain a SelectedPlan.
  2. Execute it once in STRICT mode to MEASURE its real cost (baseline).
  3. For each budget level (relaxed / moderate / strict, defined as
     fractions of that measured cost) run:
        - STRICT policy   : no intervention (records budget violation)
        - ADAPTIVE policy : monitored; on projected overrun the job group is
                            cancelled, then replan (cheaper candidate) or
                            degrade to sampling (AQP)
  4. Save one CSV row per run, including execution counts from the
     execution log (evidence for "executed exactly once").

The budget overrides the SelectedPlan's budget at runtime so this isolates
RUNTIME behaviour from planning-time feasibility.

Run:  python run_adaptive_policy_experiment.py
Tune SIZES upward if the baseline finishes too fast for the monitor
(min_elapsed_s) to ever act.

Reported numbers are local model-derived costs, not cloud billing.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import statistics
import time

from pyspark.sql import SparkSession

from mocap.cost.learned import CalibrationModel
from mocap.execution.adaptive import AdaptiveConfig, AdaptiveController
from mocap.execution.executor import SparkExecutor
from mocap.execution.replanner import make_cost_replanner
from mocap.pipeline import MOCAPPipeline
from mocap.query.parser import QueryRequest


MODEL_PATH = "results/end_to_end_calibration_model.json"
OUTPUT = "results/adaptive_policy_experiment.csv"
LOG_PATH = "results/execution_log.jsonl"

SIZES = (100_000, 1_000_000, 10_000_000)   # n, c, o row counts
NUM_CORES = 2

# Methodology: the first run pays JVM warm-up, so it is discarded.
WARMUP_RUNS = 1
BASELINE_RUNS = 3      # baseline cost = median of these
REPS = 3                # repetitions per scenario (set 1 for a quick check)
TOLERANCES = (1.0, 1.5)  # adaptive overrun_tolerance values to compare
CONFIGS = [("strict", None)] + [("adaptive", t) for t in TOLERANCES]

# Fractions of the MEASURED strict-run cost.
BUDGET_LEVELS = {
    "relaxed": 2.0,
    "moderate": 0.75,
    "strict": 0.25,
}

FIELDNAMES = [
    "scenario", "policy", "tolerance", "rep", "budget", "baseline_cost",
    "adaptive_action", "final_plan_id", "final_mode",
    "actual_cost", "cancelled_partial_cost", "total_cost",
    "wall_time_s", "within_budget",
    "sample_fraction", "accuracy_error_pct",
    "log_rows", "ok_rows", "cancelled_rows", "statuses",
]


def make_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master(f"local[{NUM_CORES}]")
        .appName("MOCAPAdaptivePolicy")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def create_query(spark: SparkSession) -> QueryRequest:
    n_rows, c_rows, o_rows = SIZES

    spark.range(0, n_rows).toDF("n_id").createOrReplaceTempView("adp_n")
    spark.range(0, c_rows).toDF("c_id").createOrReplaceTempView("adp_c")
    spark.range(0, o_rows).toDF("o_id").createOrReplaceTempView("adp_o")

    return QueryRequest(
        query_id="ADAPTIVE_POLICY_Q1",
        sql="""
            SELECT *
            FROM adp_n n
            JOIN adp_c c ON n.n_id = c.c_id
            JOIN adp_o o ON c.c_id = o.o_id
        """,
        budget=1e9,          # planning must not filter anything out;
        deadline=None,       # runtime budgets are applied later per scenario
        accuracy_tolerance=0.0,
    )


def read_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_scenario(
    spark, executor, planned, label, policy, tolerance, rep, budget,
    baseline_cost,
):
    plan = dataclasses.replace(planned.selected_plan, budget=budget)
    log_start = len(read_log(LOG_PATH))

    t0 = time.time()

    if policy == "strict":
        result = executor.execute(plan, mode="strict", split="eval")
    else:
        controller = AdaptiveController(
            executor,
            spark=spark,
            config=AdaptiveConfig(
                poll_interval_s=0.5,
                overrun_tolerance=tolerance,
                min_elapsed_s=1.0,
                min_progress=0.05,
                measure_accuracy=True,   # evaluation-only extra counts
            ),
            replanner=make_cost_replanner(planned.candidates, planned.metrics),
        )
        result = controller.run(plan)

    wall = time.time() - t0

    new_rows = read_log(LOG_PATH)[log_start:]
    statuses = [r["status"] for r in new_rows]

    rm = result.runtime_metrics
    total_cost = rm.get("total_cost_including_cancelled", result.actual_cost)

    return {
        "scenario": label,
        "policy": policy,
        "tolerance": "" if tolerance is None else tolerance,
        "rep": rep,
        "budget": budget,
        "baseline_cost": baseline_cost,
        "adaptive_action": rm.get("adaptive_action", "none"),
        "final_plan_id": result.plan_id,
        "final_mode": result.execution_mode,
        "actual_cost": result.actual_cost,
        "cancelled_partial_cost": rm.get("cancelled_partial_cost", 0.0),
        "total_cost": total_cost,
        "wall_time_s": wall,
        "within_budget": int(total_cost <= budget),
        "sample_fraction": rm.get("sample_fraction", ""),
        "accuracy_error_pct": (
            "" if rm.get("accuracy_error") is None else rm["accuracy_error"]
        ),
        "log_rows": len(new_rows),
        "ok_rows": statuses.count("ok"),
        "cancelled_rows": statuses.count("cancelled"),
        "statuses": ";".join(statuses),
    }


def main() -> None:
    spark = make_spark()

    try:
        model = (
            CalibrationModel.load(MODEL_PATH)
            if os.path.exists(MODEL_PATH)
            else None
        )

        query = create_query(spark)

        pipeline = MOCAPPipeline(
            spark=spark,
            calibration_model=model,
            num_cores=NUM_CORES,
        )
        planned = pipeline.run(query)

        if planned.selected_plan is None:
            print(f"Planning failed: {planned.failure_reason}")
            print(f"candidates generated: {len(planned.candidates)}")
            for pid, m in planned.metrics.items():
                print(
                    f"  {pid}: est_cost={m.estimated_cost:.8f} "
                    f"est_latency={m.estimated_latency:.3f}s"
                )
            raise SystemExit(1)

        executor = SparkExecutor(
            spark,
            log_path=LOG_PATH,
            num_cores=NUM_CORES,
            allow_simulation=False,
        )

        print("\nWarming up and measuring strict baseline...")
        for _ in range(WARMUP_RUNS):
            executor.execute(
                planned.selected_plan, mode="strict", split="warmup"
            )

        baseline_runs = [
            executor.execute(
                planned.selected_plan, mode="strict", split="eval"
            )
            for _ in range(BASELINE_RUNS)
        ]
        baseline_cost = statistics.median(r.actual_cost for r in baseline_runs)
        print(
            f"baseline: plan={baseline_runs[0].plan_id} "
            f"median cost={baseline_cost:.8f} "
            f"costs={[round(r.actual_cost, 8) for r in baseline_runs]} "
            f"latencies={[round(r.actual_latency, 2) for r in baseline_runs]}"
        )

        rows = []

        print("\n" + "=" * 90)
        print("MOCAP EXECUTION POLICIES")
        print("=" * 90)

        for label, factor in BUDGET_LEVELS.items():
            budget = baseline_cost * factor

            for policy, tol in CONFIGS:
                for rep_i in range(REPS):
                    row = run_scenario(
                        spark, executor, planned,
                        label, policy, tol, rep_i, budget, baseline_cost,
                    )
                    rows.append(row)

                    print(
                        f"{label:9s} {policy:8s} tol={str(tol):4s} "
                        f"rep={rep_i} budget={budget:.8f} "
                        f"action={row['adaptive_action']:9s} "
                        f"total={row['total_cost']:.8f} "
                        f"ok={row['within_budget']} "
                        f"log={row['statuses']}"
                    )

        os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

        with open(OUTPUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved results to {OUTPUT}")
        print(f"Execution log: {LOG_PATH}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
