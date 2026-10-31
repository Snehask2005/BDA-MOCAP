from pathlib import Path

import sqlglot
from sqlglot import expressions as exp
from pyspark.sql import SparkSession


FULL = Path("benchmarks/tpcds/full_sf10/parquet")
E3 = Path("benchmarks/tpcds/e3/parquet")
QUERIES = Path("benchmarks/tpcds/e3/queries")

TABLE_PATHS = {
    "store_sales": FULL / "store_sales",
    "item": FULL / "item",
    "store": FULL / "store",
    "date_dim": FULL / "date_dim",
    "customer_demographics": FULL / "customer_demographics",
    "customer": E3 / "customer",
    "customer_address": E3 / "customer_address",
    "household_demographics": E3 / "household_demographics",
    "time_dim": E3 / "time_dim",
    "promotion": E3 / "promotion",
}


def main() -> None:
    missing_paths = [
        str(path)
        for path in TABLE_PATHS.values()
        if not path.is_dir()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing table directories:\n" + "\n".join(missing_paths)
        )

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-Query-Validation")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    failures = []

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(path.resolve().as_uri()).createOrReplaceTempView(
                table
            )
            print(f"Registered {table}")

        query_files = sorted(QUERIES.glob("q*.sql"))
        if len(query_files) != 8:
            raise RuntimeError(
                f"Expected 8 query files, found {len(query_files)}"
            )

        print("\n===== E3 SPARK VALIDATION =====")

        for query_path in query_files:
            sql = query_path.read_text(encoding="utf-8")
            tree = sqlglot.parse_one(sql)
            expected_tables = int(query_path.name[1:3])
            actual_tables = len(list(tree.find_all(exp.Table)))
            actual_joins = len(tree.args.get("joins") or [])

            try:
                if actual_tables != expected_tables:
                    raise ValueError(
                        f"expected {expected_tables} tables, "
                        f"found {actual_tables}"
                    )

                if actual_joins != expected_tables - 1:
                    raise ValueError(
                        f"expected {expected_tables - 1} joins, "
                        f"found {actual_joins}"
                    )

                dataframe = spark.sql(sql)

                # Force analysis and physical-plan construction only.
                analyzed = dataframe._jdf.queryExecution().analyzed()
                physical = dataframe._jdf.queryExecution().executedPlan()

                if analyzed is None or physical is None:
                    raise RuntimeError("Spark returned an empty plan")

                print(
                    f"PASS {query_path.name}: "
                    f"{actual_tables} tables, {actual_joins} joins; "
                    "Spark analysis and physical planning succeeded"
                )

            except Exception as exc:
                failures.append((query_path.name, str(exc)))
                print(f"FAIL {query_path.name}: {exc}")

        print(f"\nValidated: {8 - len(failures)}/8 queries")

        if failures:
            print("\nFailures:")
            for name, error in failures:
                print(f"- {name}: {error}")
            raise RuntimeError("E3 query validation failed")

        print("RESULT: ALL QUERIES PASSED")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
