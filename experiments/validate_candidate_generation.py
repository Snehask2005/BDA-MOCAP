import csv
import itertools
from pathlib import Path

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates


OUTPUT = Path("results/candidate_generation_validation.csv")


def create_tables(spark):
    spark.range(100).withColumnRenamed(
        "id", "t1_id"
    ).createOrReplaceTempView("t1")

    spark.range(200).withColumnRenamed(
        "id", "t2_id"
    ).createOrReplaceTempView("t2")

    spark.range(300).withColumnRenamed(
        "id", "t3_id"
    ).createOrReplaceTempView("t3")

    spark.range(400).withColumnRenamed(
        "id", "t4_id"
    ).createOrReplaceTempView("t4")


def build_query(num_tables):
    aliases = [f"t{i}" for i in range(1, num_tables + 1)]

    select_clause = "SELECT *"

    from_clause = f"FROM {aliases[0]}"

    joins = []

    for i in range(1, num_tables):
        left = aliases[i - 1]
        right = aliases[i]

        joins.append(
            f"JOIN {right} "
            f"ON {left}.{left}_id = {right}.{right}_id"
        )

    return (
        f"{select_clause} "
        f"{from_clause} "
        + " ".join(joins)
    )


def validate_candidates(candidates):
    plan_ids = [candidate.plan_id for candidate in candidates]
    fingerprints = [candidate.fingerprint for candidate in candidates]

    unique_plan_ids = len(set(plan_ids))
    unique_fingerprints = len(set(fingerprints))

    strategies = sorted(
        set(candidate.strategy for candidate in candidates)
    )

    join_orders = sorted(
        set(
            candidate.strategy
            for candidate in candidates
            if candidate.strategy.startswith("join_order_")
        )
    )

    physical_plans = sorted(
        set(
            candidate.physical_plan
            for candidate in candidates
        )
    )

    return {
        "candidate_count": len(candidates),
        "unique_plan_ids": unique_plan_ids,
        "unique_fingerprints": unique_fingerprints,
        "strategy_count": len(strategies),
        "join_order_variant_count": len(join_orders),
        "physical_plan_count": len(physical_plans),
        "strategies": "|".join(strategies),
        "join_order_variants": "|".join(join_orders),
    }


def main():

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-Candidate-Generation-Validation")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    create_tables(spark)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("\n" + "=" * 100)
    print("MOCAP CANDIDATE GENERATION VALIDATION")
    print("=" * 100)

    for num_tables in (2, 3, 4):

        query_sql = build_query(num_tables)

        query = QueryRequest(
            query_id=f"CANDGEN_{num_tables}T",
            sql=query_sql,
            budget=100.0,
            deadline=100.0,
            accuracy_tolerance=0.0,
        )

        candidates = generate_join_candidates(
            query=query,
            spark=spark,
        )

        validation = validate_candidates(candidates)

        row = {
            "num_tables": num_tables,
            "sql": " ".join(query_sql.split()),
            **validation,
        }

        rows.append(row)

        print("\n" + "-" * 100)
        print(f"{num_tables}-TABLE QUERY")
        print("-" * 100)

        print(f"SQL:\n{query_sql}\n")

        print(
            f"Candidates:              "
            f"{validation['candidate_count']}"
        )

        print(
            f"Unique plan IDs:         "
            f"{validation['unique_plan_ids']}"
        )

        print(
            f"Unique fingerprints:     "
            f"{validation['unique_fingerprints']}"
        )

        print(
            f"Unique physical plans:   "
            f"{validation['physical_plan_count']}"
        )

        print(
            f"Strategy count:          "
            f"{validation['strategy_count']}"
        )

        print(
            f"Join-order variants:     "
            f"{validation['join_order_variant_count']}"
        )

        print("\nStrategies:")

        for strategy in validation["strategies"].split("|"):
            print(f"  - {strategy}")

        print("\nJoin-order variants:")

        if validation["join_order_variants"]:
            for variant in validation[
                "join_order_variants"
            ].split("|"):
                print(f"  - {variant}")
        else:
            print("  none")

        # -----------------------------------------------------
        # Candidate-level checks
        # -----------------------------------------------------

        assert len(candidates) > 0, (
            f"No candidates generated for "
            f"{num_tables}-table query"
        )

        assert validation["unique_plan_ids"] == len(candidates), (
            f"Duplicate plan IDs detected for "
            f"{num_tables}-table query"
        )

        assert validation["unique_fingerprints"] == len(candidates), (
            f"Duplicate fingerprints detected for "
            f"{num_tables}-table query"
        )

        # Every candidate should have a physical plan.
        for candidate in candidates:
            assert candidate.physical_plan, (
                f"Missing physical plan for "
                f"{candidate.plan_id}"
            )

            assert candidate.fingerprint, (
                f"Missing fingerprint for "
                f"{candidate.plan_id}"
            )

    with OUTPUT.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 100)
    print("VALIDATION COMPLETE")
    print("=" * 100)

    print(f"\nResults saved to:")
    print(OUTPUT)

    spark.stop()


if __name__ == "__main__":
    main()
