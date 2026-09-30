import csv
import time
from pathlib import Path

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates
from mocap.cost.statistics import extract_plan_statistics
from mocap.cost.predictor import predict_candidates, PredictorConfig
from mocap.cost.estimator import observe_execution_telemetry


def main():

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-Candidate-Execution-Validation")
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
        query_id="candidate_execution_validation",
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
    print("MOCAP CANDIDATE EXECUTION VALIDATION")
    print("=" * 100)

    print(f"\nCandidates generated: {len(candidates)}")

    # ---------------------------------------------------------
    # 4. Evaluate every candidate
    # ---------------------------------------------------------

    results = []

    for index, candidate in enumerate(candidates, start=1):

        print(
            f"\n[{index}/{len(candidates)}] "
            f"{candidate.plan_id} | "
            f"{candidate.strategy}"
        )

        dataframe = spark.sql(candidate.sql)

        # ---------------------------------------------
        # Candidate-specific statistics
        # ---------------------------------------------

        statistics = extract_plan_statistics(dataframe)

        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        predicted = predict_candidates(
            [candidate],
            config=predictor_config,
            statistics=statistics,
        )

        metrics = predicted[candidate.plan_id]

        # ---------------------------------------------
        # Execute candidate
        # ---------------------------------------------

        start = time.perf_counter()

        row_count = dataframe.count()

        elapsed = time.perf_counter() - start

        # ---------------------------------------------
        # Observe execution telemetry
        # ---------------------------------------------

        telemetry = observe_execution_telemetry(
            candidate=candidate,
            dataframe=dataframe,
            elapsed=elapsed,
            num_cores=2,
        )

        result = {
            "plan_id": candidate.plan_id,
            "strategy": candidate.strategy,
            "joins": candidate.num_joins,
            "broadcasts": candidate.num_broadcast_joins,
            "shuffle_hash": candidate.num_shuffle_hash_joins,
            "sort_merge": candidate.num_sort_merge_joins,
            "exchanges": candidate.num_exchanges,
            "input_bytes": statistics.input_bytes,
            "intermediate_bytes": statistics.intermediate_bytes,
            "row_count_stat": statistics.row_count,
            "predicted_cost": metrics.estimated_cost,
            "predicted_latency": metrics.estimated_latency,
            "actual_latency_seconds": elapsed,
            "actual_cost": telemetry.actual_cost,
            "cpu_seconds": telemetry.cpu_seconds,
            "bytes_scanned": telemetry.bytes_scanned,
            "bytes_shuffled": telemetry.bytes_shuffled,
            "result_row_count": row_count,
            "fingerprint": candidate.fingerprint,
        }

        results.append(result)

        print(
            f"    predicted cost    = {metrics.estimated_cost:.8f}"
        )
        print(
            f"    predicted latency = {metrics.estimated_latency:.6f}s"
        )
        print(
            f"    actual latency    = {elapsed:.6f}s"
        )
        print(
            f"    actual cost       = {telemetry.actual_cost:.8f}"
        )
        print(
            f"    result rows       = {row_count}"
        )

    # ---------------------------------------------------------
    # 5. Save results
    # ---------------------------------------------------------

    output_dir = Path(
        "results/candidate_execution_validation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir /
        "candidate_execution_results.csv"
    )

    with output_file.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    # ---------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 100)
    print("EXECUTION VALIDATION COMPLETE")
    print("=" * 100)

    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )

    print("\nCandidate summary:")

    for result in results:

        print(
            f"{result['plan_id']:<55} "
            f"pred={result['predicted_latency']:.4f}s "
            f"actual={result['actual_latency_seconds']:.4f}s"
        )

    print("\n" + "=" * 100)

    spark.stop()


if __name__ == "__main__":
    main()
