from pyspark.sql import SparkSession
import csv
from pathlib import Path
from mocap.plans.generator import generate_join_candidates
from mocap.cost.statistics import extract_plan_statistics
from mocap.cost.predictor import predict_candidates, PredictorConfig
from mocap.query.parser import QueryRequest

def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-Candidate-Specific-Validation")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # ---------------------------------------------------------
    # 1. Create deliberately asymmetric relations
    # ---------------------------------------------------------

    n = spark.range(0, 100).withColumnRenamed("id", "n_id")

    c = (
        spark.range(0, 1000)
        .withColumnRenamed("id", "c_id")
    )

    o = (
        spark.range(0, 10000)
        .withColumnRenamed("id", "o_id")
    )

    n.createOrReplaceTempView("n")
    c.createOrReplaceTempView("c")
    o.createOrReplaceTempView("o")

    # ---------------------------------------------------------
    # 2. Query with asymmetric join selectivity
    # ---------------------------------------------------------

    sql = """
    SELECT *
    FROM n
    JOIN c ON n.n_id = c.c_id
    JOIN o ON c.c_id = o.o_id
    """

    print("\n" + "=" * 80)
    print("MOCAP CANDIDATE-SPECIFIC STATISTICS VALIDATION")
    print("=" * 80)

    print("\nBase query:")
    print(sql.strip())

    # ---------------------------------------------------------
    # 3. Generate candidate physical plans
   # ---------------------------------------------------------


    query = QueryRequest(
        query_id="candidate_specific_validation",
        sql=sql,
        budget=100.0,
        deadline=100.0,
        accuracy_tolerance=0.0,
    )

    candidates = generate_join_candidates(
        query=query,
        spark=spark,
    )
    print(f"\nNumber of generated candidates: {len(candidates)}")
    if not candidates:
        print("ERROR: No candidates generated.")
        spark.stop()
        return

    # ---------------------------------------------------------
    # 4. Extract statistics independently for EACH candidate
    # ---------------------------------------------------------

    config = PredictorConfig()

    results = []

    for candidate in candidates:

        dataframe = spark.sql(candidate.sql)

        statistics = extract_plan_statistics(dataframe)

        predicted = predict_candidates(
            [candidate],
            config=config,
            statistics=statistics,
        )

        metrics = predicted[candidate.plan_id]

        results.append(
            {
                "plan_id": candidate.plan_id,
                "strategy": candidate.strategy,
                "joins": candidate.num_joins,
                "broadcasts": candidate.num_broadcast_joins,
                "shuffle_hash": candidate.num_shuffle_hash_joins,
                "sort_merge": candidate.num_sort_merge_joins,
                "exchanges": candidate.num_exchanges,
                "input_bytes": statistics.input_bytes,
                "intermediate_bytes": statistics.intermediate_bytes,
                "row_count": statistics.row_count,
                "predicted_cost": metrics.estimated_cost,
                "predicted_latency": metrics.estimated_latency,
                "fingerprint": candidate.fingerprint,
            }
        )
     # ---------------------------------------------------------
    # 5. Save machine-readable experiment results
    # ---------------------------------------------------------

    output_dir = Path("results/candidate_specific_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "candidate_predictions.csv"

    fieldnames = list(results[0].keys())

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved experiment results to: {output_file}")
    # ---------------------------------------------------------
    # 5. Print experiment table
    # ---------------------------------------------------------

    print("\n" + "-" * 120)
    print(
        f"{'PLAN':<7}"
        f"{'STRATEGY':<14}"
        f"{'JOIN':<6}"
        f"{'BCAST':<7}"
        f"{'SHUFFLE':<8}"
        f"{'SMJ':<6}"
        f"{'INPUT':<14}"
        f"{'INTERMEDIATE':<16}"
        f"{'ROWS':<10}"
        f"{'COST':<14}"
        f"{'LATENCY':<12}"
    )
    print("-" * 120)

    for r in results:
        print(
            f"{r['plan_id']:<7}"
            f"{r['strategy']:<14}"
            f"{r['joins']:<6}"
            f"{r['broadcasts']:<7}"
            f"{r['shuffle_hash']:<8}"
            f"{r['sort_merge']:<6}"
            f"{r['input_bytes']:<14.2f}"
            f"{r['intermediate_bytes']:<16.2f}"
            f"{str(r['row_count']):<10}"
            f"{r['predicted_cost']:<14.8f}"
            f"{r['predicted_latency']:<12.6f}"
        )

    print("-" * 120)

    # ---------------------------------------------------------
    # 6. Check whether candidate-specific statistics vary
    # ---------------------------------------------------------

    intermediate_values = {
        r["intermediate_bytes"]
        for r in results
    }

    cost_values = {
        round(r["predicted_cost"], 12)
        for r in results
    }

    latency_values = {
        round(r["predicted_latency"], 12)
        for r in results
    }

    print("\nValidation:")
    print(f"Unique intermediate-byte values : {len(intermediate_values)}")
    print(f"Unique predicted-cost values    : {len(cost_values)}")
    print(f"Unique predicted-latency values : {len(latency_values)}")

    if len(intermediate_values) > 1:
        print("✓ Candidate-specific intermediate statistics differ.")
    else:
        print("⚠ Intermediate statistics are identical.")

    if len(cost_values) > 1:
        print("✓ Candidate-specific predicted costs differ.")
    else:
        print("⚠ Predicted costs are identical.")

    if len(latency_values) > 1:
        print("✓ Candidate-specific predicted latencies differ.")
    else:
        print("⚠ Predicted latencies are identical.")

    print("\n" + "=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()
