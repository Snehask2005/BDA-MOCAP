import sqlglot


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
    tree = sqlglot.parse_one(SQL)

    print("=" * 70)
    print("SQL AST")
    print("=" * 70)

    print(tree)

    print("\nTables:")

    for table in tree.find_all(sqlglot.exp.Table):
        print(
            f"  name={table.name}, alias={table.alias}"
        )

    print("\nJoins:")

    for join in tree.find_all(sqlglot.exp.Join):
        print(join)


if __name__ == "__main__":
    main()