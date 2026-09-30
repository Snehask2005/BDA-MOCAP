import csv
import random
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates
from mocap.cost.statistics import extract_plan_statistics
from mocap.cost.predictor import predict_candidates, PredictorConfig
from mocap.cost.estimator import observe_execution_telemetry


REPETITIONS = 3


def execute_candidate(candidate, spark, num_cores=2):
    """
    Execute one candidate once and return measured latency
    and telemetry.
    """

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
        .appName("MOCAP-Repeated-Candidate-Execution")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # ---------------------------------------------------------
    # 1. Create asymmetric relations
    # ---------------------------------------------------------

    spark.range(100).withColumnRenamed(
        "id", "n_id"
    ).createOrReplaceTempView("n")

    spark.range(1000).withColumnRenamed(
        "id", "c_id"
    ).createOrReplaceTempView("c")

    spark.range(10000).withColumnRenamed(
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
        query_id="repeated_candidate_execution",
        sql=sql,
        budget=100.0,
        deadline=100.0,
        accuracy_tolerance=0.0,
    )

    predictor_config = PredictorConfig()

    # ---------------------------------------------------------
    # 3. Generate candidates
    # ---------------------------------------------------------

    candidates = generate_join_candidates(
        query=query,
        spark=spark,
    )

    print("\n" + "=" * 100)
    print("MOCAP REPEATED CANDIDATE EXECUTION")
    print("=" * 100)

    print(f"\nCandidates: {len(candidates)}")
    print(f"Measured repetitions per candidate: {REPETITIONS}")

    # ---------------------------------------------------------
    # 4. Compute candidate-specific predictions once
    # ---------------------------------------------------------

    candidate_info = {}

    for candidate in candidates:

        dataframe = spark.sql(candidate.sql)

        statistics = extract_plan_statistics(dataframe)

        predicted = predict_candidates(
            [candidate],
            config=predictor_config,
            statistics=statistics,
        )

        metrics = predicted[candidate.plan_id]

        candidate_info[candidate.plan_id] = {
            "candidate": candidate,
            "input_bytes": statistics.input_bytes,
            "intermediate_bytes": statistics.intermediate_bytes,
            "row_count_stat": statistics.row_count,
            "predicted_cost": metrics.estimated_cost,
            "predicted_latency": metrics.estimated_latency,
        }

    # ---------------------------------------------------------
    # 5. Warm-up
    # ---------------------------------------------------------

    print("\nWarm-up phase...")

    warmup_order = candidates.copy()
    random.Random(42).shuffle(warmup_order)

    for candidate in warmup_order:

        _ = execute_candidate(
            candidate,
            spark,
            num_cores=2,
        )

    print("Warm-up complete.")

    # ---------------------------------------------------------
    # 6. Repeated randomized measurements
    # ---------------------------------------------------------

    measured_rows = []

    for repetition in range(1, REPETITIONS + 1):

        execution_order = candidates.copy()

        # Deterministic randomization makes the experiment reproducible.
        random.Random(1000 + repetition).shuffle(
            execution_order
        )

        print(
            f"\n{'=' * 80}\n"
            f"MEASUREMENT REPETITION {repetition}/{REPETITIONS}\n"
            f"{'=' * 80}"
        )

        for position, candidate in enumerate(
            execution_order,
            start=1,
        ):

            result = execute_candidate(
                candidate,
                spark,
                num_cores=2,
            )

            info = candidate_info[candidate.plan_id]

            measured_rows.append(
                {
                    "repetition": repetition,
                    "execution_position": position,
                    "plan_id": candidate.plan_id,
                    "strategy": candidate.strategy,
                    "joins": candidate.num_joins,
                    "broadcasts": candidate.num_broadcast_joins,
                    "shuffle_hash": candidate.num_shuffle_hash_joins,
                    "sort_merge": candidate.num_sort_merge_joins,
                    "exchanges": candidate.num_exchanges,
                    "input_bytes": info["input_bytes"],
                    "intermediate_bytes": info["intermediate_bytes"],
                    "row_count_stat": info["row_count_stat"],
                    "predicted_cost": info["predicted_cost"],
                    "predicted_latency": info["predicted_latency"],
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
                f"{position:02d}. "
                f"{candidate.plan_id:<50} "
                f"{result['latency']:.4f}s"
            )

    # ---------------------------------------------------------
    # 7. Save per-run results
    # ---------------------------------------------------------

    output_dir = Path(
        "results/candidate_repeated_execution"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_file = output_dir / "repeated_execution_raw.csv"

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
    # 8. Calculate candidate-level summary
    # ---------------------------------------------------------

    grouped = {}

    for row in measured_rows:

        grouped.setdefault(
            row["plan_id"],
            [],
        ).append(row)

    summary_rows = []

    for plan_id, rows in grouped.items():

        latencies = [
            row["actual_latency_seconds"]
            for row in rows
        ]

        costs = [
            row["actual_cost"]
            for row in rows
        ]

        first = rows[0]

        summary_rows.append(
            {
                "plan_id": plan_id,
                "strategy": first["strategy"],
                "joins": first["joins"],
                "broadcasts": first["broadcasts"],
                "shuffle_hash": first["shuffle_hash"],
                "sort_merge": first["sort_merge"],
                "exchanges": first["exchanges"],
                "input_bytes": first["input_bytes"],
                "intermediate_bytes": first["intermediate_bytes"],
                "row_count_stat": first["row_count_stat"],
                "predicted_cost": first["predicted_cost"],
                "predicted_latency": first["predicted_latency"],
                "actual_latency_mean": statistics.mean(latencies),
                "actual_latency_median": statistics.median(latencies),
                "actual_latency_min": min(latencies),
                "actual_latency_max": max(latencies),
                "actual_cost_mean": statistics.mean(costs),
                "actual_cost_median": statistics.median(costs),
                "fingerprint": first["fingerprint"],
            }
        )

    summary_rows.sort(
        key=lambda row: row["actual_latency_median"]
    )

    summary_file = (
        output_dir /
        "candidate_execution_summary.csv"
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

    # ---------------------------------------------------------
    # 9. Print summary
    # ---------------------------------------------------------

    print("\n" + "=" * 120)
    print("CANDIDATE EXECUTION SUMMARY")
    print("=" * 120)

    print(
        f"{'PLAN':<52}"
        f"{'PRED':>10}"
        f"{'MEDIAN':>12}"
        f"{'MIN':>10}"
        f"{'MAX':>10}"
    )

    print("-" * 120)

    for row in summary_rows:

        print(
            f"{row['plan_id']:<52}"
            f"{row['predicted_latency']:>10.4f}"
            f"{row['actual_latency_median']:>12.4f}"
            f"{row['actual_latency_min']:>10.4f}"
            f"{row['actual_latency_max']:>10.4f}"
        )

    print("-" * 120)

    print(
        "\nRaw results saved to:\n"
        f"  {raw_file}"
    )

    print(
        "\nSummary saved to:\n"
        f"  {summary_file}"
    )

    print("\n" + "=" * 120)

    spark.stop()


if __name__ == "__main__":
    main()
