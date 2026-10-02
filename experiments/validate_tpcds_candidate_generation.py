import csv
import time
from pathlib import Path

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates
from mocap.pipeline import MOCAPPipeline


BASE = "benchmarks/tpcds/representative/parquet"
SQL_PATH = (
    "benchmarks/tpcds/representative/queries/"
    "q5_join_order.sql"
)

OUTPUT = Path(
    "results/tpcds_candidate_generation_validation.csv"
)


def load_tables(spark):
    tables = [
        "store_sales",
        "item",
        "store",
        "date_dim",
        "customer_demographics",
    ]

    for table in tables:
        df = spark.read.parquet(f"{BASE}/{table}")
        df.createOrReplaceTempView(table)
        print(f"{table}: {df.count()} rows")


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-TPCDS-Candidate-Generation")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    load_tables(spark)

    with open(SQL_PATH) as f:
        sql = f.read().strip()

    query = QueryRequest(
        query_id="TPCDS_DERIVED_5T",
        sql=sql,
        budget=100.0,
        deadline=100.0,
        accuracy_tolerance=0.0,
    )

    print("\n" + "=" * 100)
    print("MOCAP — TPC-DS-DERIVED 5-TABLE CANDIDATE VALIDATION")
    print("=" * 100)

    print("\nQuery:")
    print(sql)

    # ---------------------------------------------------------
    # Candidate generation
    # ---------------------------------------------------------

    start = time.perf_counter()

    candidates = generate_join_candidates(
        query=query,
        spark=spark,
    )

    generation_time = time.perf_counter() - start

    print("\n" + "-" * 100)
    print("CANDIDATE GENERATION")
    print("-" * 100)

    print(f"Candidates:            {len(candidates)}")
    print(
        f"Unique plan IDs:       "
        f"{len(set(c.plan_id for c in candidates))}"
    )
    print(
        f"Unique fingerprints:   "
        f"{len(set(c.fingerprint for c in candidates))}"
    )
    print(
        f"Unique physical plans: "
        f"{len(set(c.physical_plan for c in candidates))}"
    )

    strategies = sorted(
        set(c.strategy for c in candidates)
    )

    join_orders = sorted(
        set(
            c.strategy
            for c in candidates
            if c.strategy.startswith("join_order_")
        )
    )

    print(f"Strategies:            {len(strategies)}")
    print(f"Join-order variants:   {len(join_orders)}")
    print(f"Generation time:       {generation_time:.6f} s")

    print("\nStrategies:")
    for strategy in strategies:
        print(f"  {strategy}")

    print("\nJoin-order variants:")
    for strategy in join_orders:
        print(f"  {strategy}")

    # ---------------------------------------------------------
    # Candidate details
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("CANDIDATE DETAILS")
    print("-" * 100)

    rows = []

    for candidate in candidates:
        rows.append(
            {
                "query_id": query.query_id,
                "plan_id": candidate.plan_id,
                "strategy": candidate.strategy,
                "fingerprint": candidate.fingerprint,
                "physical_plan": candidate.physical_plan,
                "sql": " ".join(candidate.sql.split()),
            }
        )

        print(
            f"{candidate.plan_id} | "
            f"{candidate.strategy} | "
            f"fingerprint={candidate.fingerprint}"
        )

    # ---------------------------------------------------------
    # Candidate-specific estimation through MOCAP
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("CANDIDATE-SPECIFIC ESTIMATION")
    print("-" * 100)

    pipeline = MOCAPPipeline(
        spark=spark,
        num_cores=2,
    )

    start = time.perf_counter()

    metrics = pipeline.estimate_candidates(candidates)

    estimation_time = time.perf_counter() - start

    print(f"Estimation time: {estimation_time:.6f} s")

    for candidate in candidates:
        metric = metrics[candidate.plan_id]

        print(
            f"{candidate.plan_id} | "
            f"{candidate.strategy} | "
            f"cost={metric.estimated_cost:.10f} | "
            f"latency={metric.estimated_latency:.6f} | "
            f"scanned={metric.bytes_scanned:.2f} | "
            f"shuffled={metric.bytes_shuffled:.2f}"
        )

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert candidates, "No candidates generated."

    assert len(set(c.plan_id for c in candidates)) == len(candidates), (
        "Duplicate plan IDs detected."
    )

    assert len(set(c.fingerprint for c in candidates)) == len(candidates), (
        "Duplicate fingerprints detected."
    )

    assert all(c.physical_plan for c in candidates), (
        "At least one candidate has no physical plan."
    )

    assert len(join_orders) > 0, (
        "No join-order variants generated for 5-table query."
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nResults saved to:")
    print(OUTPUT)

    print("\n" + "=" * 100)
    print("TPC-DS-DERIVED CANDIDATE VALIDATION COMPLETE")
    print("=" * 100)

    spark.stop()


if __name__ == "__main__":
    main()
