from mocap.query.parser import extract_query_structure


SQL = """
SELECT
    c.c_mktsegment,
    COUNT(*) AS order_count
FROM customer c
JOIN orders o
    ON c.c_custkey = o.o_custkey
WHERE o.o_totalprice > 1000
GROUP BY c.c_mktsegment;
"""


def main() -> None:
    structure = extract_query_structure(SQL)

    print("=" * 70)
    print("QUERY STRUCTURE")
    print("=" * 70)

    print("\nTables:")

    for table in structure.tables:
        print(
            f"  name={table.name}, alias={table.alias}"
        )

    print("\nJoins:")

    for join in structure.joins:
        print(f"  type={join.join_type}")
        print(
            f"  table={join.table.name}"
        )
        print(
            f"  alias={join.table.alias}"
        )
        print(
            f"  condition={join.condition}"
        )


if __name__ == "__main__":
    main()