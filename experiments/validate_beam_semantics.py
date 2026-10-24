from pyspark.sql import SparkSession

from mocap.plans.generator import _generate_join_order_variants
from mocap.query.parser import QueryRequest


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-Beam-Semantics")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        datasets = {
            "table_a": (
                [(1,), (2,)],
                ["id"],
            ),
            "table_b": (
                [(10, 1), (20, 2)],
                ["id", "a_id"],
            ),
            "table_c": (
                [(100, 10), (200, 20)],
                ["id", "b_id"],
            ),
            "table_d": (
                [(1000, 100), (2000, 200)],
                ["id", "c_id"],
            ),
            "table_e": (
                [(10000, 1000), (20000, 2000)],
                ["id", "d_id"],
            ),
            "table_f": (
                [(100000, 10000), (200000, 20000)],
                ["id", "e_id"],
            ),
        }

        for name, (rows, columns) in datasets.items():
            spark.createDataFrame(
                rows, columns
            ).createOrReplaceTempView(name)

        sql = """
        SELECT a.id AS a_id, f.id AS f_id
        FROM table_a a
        JOIN table_b b ON a.id = b.a_id
        JOIN table_c c ON b.id = c.b_id
        JOIN table_d d ON c.id = d.c_id
        JOIN table_e e ON d.id = e.d_id
        JOIN table_f f ON e.id = f.e_id
        """

        query = QueryRequest(
            query_id="BEAM_SEMANTICS",
            sql=sql,
            budget=100.0,
        )

        original_rows = sorted(
            tuple(row)
            for row in spark.sql(sql).collect()
        )

        variants = _generate_join_order_variants(
            query,
            search_mode="beam",
            beam_width=10,
            table_size_estimates={
                "a": 1000.0,
                "b": 100.0,
                "c": 20.0,
                "d": 5.0,
                "e": 10.0,
                "f": 500.0,
            },
        )

        print(f"Generated join-order variants: {len(variants)}")
        assert variants, "Beam search generated no complete join orders."

        for index, (label, candidate_sql) in enumerate(
            variants, start=1
        ):
            actual_rows = sorted(
                tuple(row)
                for row in spark.sql(candidate_sql).collect()
            )

            if actual_rows != original_rows:
                raise AssertionError(
                    f"Results differ for {label}\n"
                    f"Expected: {original_rows}\n"
                    f"Actual:   {actual_rows}\n"
                    f"SQL:\n{candidate_sql}"
                )

            print(f"[PASS] {index:02d} {label}")

        print(
            "\nSUCCESS: All generated join orders returned "
            "the same results as the original query."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
