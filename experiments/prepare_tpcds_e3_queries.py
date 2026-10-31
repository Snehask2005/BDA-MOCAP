from pathlib import Path

import sqlglot
from sqlglot import expressions as exp


OUT = Path("benchmarks/tpcds/e3/queries")

# Start with store_sales and progressively add connected dimensions.
# Each entry adds exactly one table and one join predicate.
JOINS = [
    (
        "date_dim d",
        "ss.ss_sold_date_sk = d.d_date_sk",
    ),
    (
        "item i",
        "ss.ss_item_sk = i.i_item_sk",
    ),
    (
        "store s",
        "ss.ss_store_sk = s.s_store_sk",
    ),
    (
        "customer_demographics cd",
        "ss.ss_cdemo_sk = cd.cd_demo_sk",
    ),
    (
        "time_dim t",
        "ss.ss_sold_time_sk = t.t_time_sk",
    ),
    (
        "customer c",
        "ss.ss_customer_sk = c.c_customer_sk",
    ),
    (
        "customer_address ca",
        "c.c_current_addr_sk = ca.ca_address_sk",
    ),
    (
        "household_demographics hd",
        "ss.ss_hdemo_sk = hd.hd_demo_sk",
    ),
    (
        "promotion p",
        "ss.ss_promo_sk = p.p_promo_sk",
    ),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    for table_count in range(3, 11):
        selected_joins = JOINS[:table_count - 1]

        lines = [
            "SELECT",
            "    COUNT(*) AS row_count,",
            "    SUM(ss.ss_ext_sales_price) AS total_sales",
            "FROM store_sales ss",
        ]

        for relation, predicate in selected_joins:
            lines.extend([
                f"INNER JOIN {relation}",
                f"    ON {predicate}",
            ])

        # Keep the shared predicate identical across every query.
        lines.extend([
            "WHERE d.d_year BETWEEN 2000 AND 2002",
            "",
        ])

        sql = "\n".join(lines)
        tree = sqlglot.parse_one(sql)

        actual_tables = sum(
            1 for _ in tree.find_all(exp.Table)
        )
        actual_joins = len(tree.args.get("joins") or [])

        if actual_tables != table_count:
            raise RuntimeError(
                f"q{table_count:02d}: expected {table_count} tables, "
                f"found {actual_tables}"
            )

        if actual_joins != table_count - 1:
            raise RuntimeError(
                f"q{table_count:02d}: unexpected join count "
                f"{actual_joins}"
            )

        path = OUT / f"q{table_count:02d}_{table_count}_tables.sql"
        path.write_text(sql, encoding="utf-8")

        print(
            f"PASS {path}: "
            f"{actual_tables} tables, {actual_joins} joins"
        )

    print(f"\nCreated 8 validated SQL queries under {OUT}")


if __name__ == "__main__":
    main()
