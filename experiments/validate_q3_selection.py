import csv
import random
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates
from mocap.cost.estimator import observe_execution_telemetry


REPETITIONS = 3
TARGET_PLAN_SUFFIXES = {"_P02", "_P05"}


def execute_candidate(candidate, spark, num_cores=2):
    dataframe = spark.sql(candidate.sql)

    start = time.perf_counter()

    row_count = dataframe.count()

    elapsed = time.perf_counter() - start

    telemetry = observe_execution_telemetry(
        candidate=candidate,
        dataframe=dataframe,
        elapsed=elapsed,
        num_cores=num_cores,
    )

    return {
        "latency": elapsed,
        "actual_cost": telemetry.actual_cost,
        "cpu_seconds": telemetry.cpu_seconds,
        "bytes_scanned": telemetry.bytes_scanned,
        "bytes_shuffled": telemetry.bytes_shuffled,
        "row_count": row_count,
    }


def main():

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-Q3-Selection-Validation")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # ---------------------------------------------------------
    # 1. Create Q3 balanced workload
    # ---------------------------------------------------------

    spark.range(1000).withColumnRenamed(
        "id", "n_id"
    ).createOrReplaceTempView("n")

    spark.range(2000).withColumnRenamed(
        "id", "c_id"
    ).createOrReplaceTempView("c")

    spark.range(3000).withColumnRenamed(
        "id", "o_id"
    ).createOrReplaceTempView("o")

    # ---------------------------------------------------------
    # 2. Query
    # ---------------------------------------------------------

    sql = """
        SELECT *
        FROM n
        JOIN c ON n.n_id = c.c_id
        JOIN o ON c.c_id = o.o_id
    """

    query = QueryRequest(
        query_id="Q3_execution_validation",
        sql=sql,
        budget=7.8295997e-05,
        deadline=100.0,
        accuracy_tolerance=0.0,
    )

    # ---------------------------------------------------------
    # 3. Generate all candidates
    # ---------------------------------------------------------

    candidates = generate_join_candidates(
        query=query,
        spark=spark,
    )

    selected = {
        candidate.plan_id: candidate
        for candidate in candidates
        if any(
            candidate.plan_id.endswith(suffix)
            for suffix in TARGET_PLAN_SUFFIXES
        )
    }

    found_suffixes = {
        suffix
        for suffix in TARGET_PLAN_SUFFIXES
        if any(
            candidate.plan_id.endswith(suffix)
            for candidate in candidates
        )
    }

    missing = TARGET_PLAN_SUFFIXES - found_suffixes

    if missing:
        raise RuntimeError(
            f"Could not find expected plan suffixes: {sorted(missing)}"
        )

    print("\n" + "=" * 90)
    print("MOCAP Q3 SELECTION VALIDATION")
    print("=" * 90)

    print("\nWorkload:")
    print("  n = 1,000 rows")
    print("  c = 2,000 rows")
    print("  o = 3,000 rows")

    print("\nPlans being compared:")
    print("  P02 = MOCAP-selected plan")
    print("  P05 = min-latency baseline")

    print(f"\nBudget: {query.budget}")
    print(f"Measured repetitions: {REPETITIONS}")

    for plan_id, candidate in selected.items():
        print(f"\n{plan_id}")
        print(f"  strategy: {candidate.strategy}")
        print(f"  SQL: {candidate.sql}")

    # ---------------------------------------------------------
    # 4. Warm-up
    # ---------------------------------------------------------

    print("\n" + "-" * 90)
    print("WARM-UP")
    print("-" * 90)

    warmup_order = list(selected.values())
    random.Random(42).shuffle(warmup_order)

    for candidate in warmup_order:
        result = execute_candidate(
            candidate,
            spark,
            num_cores=2,
        )

        print(
            f"{candidate.plan_id}: "
            f"{result['latency']:.4f}s"
        )

    # ---------------------------------------------------------
    # 5. Repeated measurements
    # ---------------------------------------------------------

    measured_rows = []

    for repetition in range(1, REPETITIONS + 1):

        execution_order = list(selected.values())

        random.Random(1000 + repetition).shuffle(
            execution_order
        )

        print("\n" + "-" * 90)
        print(
            f"MEASUREMENT REPETITION "
            f"{repetition}/{REPETITIONS}"
        )
        print("-" * 90)

        for position, candidate in enumerate(
            execution_order,
            start=1,
        ):

            result = execute_candidate(
                candidate,
                spark,
                num_cores=2,
            )

            measured_rows.append(
                {
                    "workload": "Q3_balanced",
                    "repetition": repetition,
                    "execution_position": position,
                    "plan_id": candidate.plan_id,
                    "strategy": candidate.strategy,
                    "actual_latency_seconds": result["latency"],
                    "actual_cost": result["actual_cost"],
                    "cpu_seconds": result["cpu_seconds"],
                    "bytes_scanned": result["bytes_scanned"],
                    "bytes_shuffled": result["bytes_shuffled"],
                    "result_row_count": result["row_count"],
                    "fingerprint": candidate.fingerprint,
                }
            )

            print(
                f"{position}. "
                f"{candidate.plan_id}: "
                f"{result['latency']:.4f}s"
            )

    # ---------------------------------------------------------
    # 6. Save raw measurements
    # ---------------------------------------------------------

    output_dir = Path(
        "results/q3_selection_validation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_file = output_dir / "q3_execution_raw.csv"

    with raw_file.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(measured_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(measured_rows)

    # ---------------------------------------------------------
    # 7. Summarize
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    summary_rows = []

    for plan_id in sorted(selected):
        rows = [
            row
            for row in measured_rows
            if row["plan_id"] == plan_id
        ]

        latencies = [
            row["actual_latency_seconds"]
            for row in rows
        ]

        costs = [
            row["actual_cost"]
            for row in rows
        ]

        median_latency = statistics.median(latencies)
        mean_latency = statistics.mean(latencies)
        median_cost = statistics.median(costs)

        summary_rows.append(
            {
                "workload": "Q3_balanced",
                "plan_id": plan_id,
                "repetitions": len(rows),
                "median_latency_seconds": median_latency,
                "mean_latency_seconds": mean_latency,
                "median_actual_cost": median_cost,
                "latency_runs": ";".join(
                    f"{value:.6f}"
                    for value in latencies
                ),
                "cost_runs": ";".join(
                    f"{value:.10f}"
                    for value in costs
                ),
            }
        )

        print(f"\n{plan_id}")
        print(
            "  Latencies: "
            + ", ".join(
                f"{value:.4f}s"
                for value in latencies
            )
        )
        print(
            f"  Median latency: "
            f"{median_latency:.4f}s"
        )
        print(
            f"  Mean latency: "
            f"{mean_latency:.4f}s"
        )
        print(
            f"  Median actual cost: "
            f"{median_cost:.10f}"
        )

    summary_file = (
        output_dir / "q3_execution_summary.csv"
    )

    with summary_file.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 90)
    print("FILES")
    print("=" * 90)

    print(raw_file)
    print(summary_file)

    spark.stop()


if __name__ == "__main__":
    main()
