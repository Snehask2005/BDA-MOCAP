"""
Experiment workloads (owner: Anaswara).

Two sources of queries:

1. ``star_join_workload()`` -- the E3 star-join family (3-10 tables, same
   joins as experiments/prepare_tpcds_e3_queries.py) crossed with three
   date-range selectivities, plus the representative Q5 group-by query.
   Every query is labelled with a *family* so E5 can look for a family
   where MOCAP beats Greedy.
2. ``load_sql_dir()`` -- any directory of ``*.sql`` files (e.g. official
   TPC-DS query text). Queries the generator cannot handle are skipped and
   reported by the harness, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

# Same join chain as experiments/prepare_tpcds_e3_queries.py.
_JOINS = [
    ("date_dim d", "ss.ss_sold_date_sk = d.d_date_sk"),
    ("item i", "ss.ss_item_sk = i.i_item_sk"),
    ("store s", "ss.ss_store_sk = s.s_store_sk"),
    ("customer_demographics cd", "ss.ss_cdemo_sk = cd.cd_demo_sk"),
    ("time_dim t", "ss.ss_sold_time_sk = t.t_time_sk"),
    ("customer c", "ss.ss_customer_sk = c.c_customer_sk"),
    ("customer_address ca", "c.c_current_addr_sk = ca.ca_address_sk"),
    ("household_demographics hd", "ss.ss_hdemo_sk = hd.hd_demo_sk"),
    ("promotion p", "ss.ss_promo_sk = p.p_promo_sk"),
]

SELECTIVITIES = {
    "narrow": "d.d_year = 2000",
    "mid": "d.d_year BETWEEN 2000 AND 2002",
    "wide": "d.d_year BETWEEN 1998 AND 2003",
}

Q5_SQL = """SELECT
    d.d_year, s.s_state, i.i_category, cd.cd_gender,
    SUM(ss.ss_ext_sales_price) AS total_sales
FROM store_sales ss
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
JOIN store s ON ss.ss_store_sk = s.s_store_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN customer_demographics cd ON ss.ss_cdemo_sk = cd.cd_demo_sk
WHERE {pred}
  AND s.s_state IS NOT NULL
  AND i.i_category IS NOT NULL
  AND cd.cd_gender IS NOT NULL
GROUP BY d.d_year, s.s_state, i.i_category, cd.cd_gender"""


@dataclass(frozen=True)
class WorkloadQuery:
    query_id: str
    sql: str
    family: str
    num_tables: int
    selectivity: Optional[str] = None


def _family_for(num_tables: int) -> str:
    if num_tables <= 4:
        return "star_small"
    if num_tables <= 7:
        return "star_medium"
    return "star_large"


def star_query_sql(num_tables: int, predicate: str) -> str:
    if not 3 <= num_tables <= len(_JOINS) + 1:
        raise ValueError("num_tables must be between 3 and 10")
    lines = [
        "SELECT",
        "    COUNT(*) AS row_count,",
        "    SUM(ss.ss_ext_sales_price) AS total_sales",
        "FROM store_sales ss",
    ]
    for relation, join_pred in _JOINS[: num_tables - 1]:
        lines.append(f"INNER JOIN {relation}")
        lines.append(f"    ON {join_pred}")
    lines.append(f"WHERE {predicate}")
    return "\n".join(lines)


def star_join_workload(
    table_counts: Sequence[int] = range(3, 11),
    selectivities: Sequence[str] = ("narrow", "mid", "wide"),
    include_q5: bool = True,
) -> List[WorkloadQuery]:
    queries: List[WorkloadQuery] = []
    for k in table_counts:
        for sel in selectivities:
            queries.append(
                WorkloadQuery(
                    query_id=f"E3_Q{k:02d}_{sel}",
                    sql=star_query_sql(k, SELECTIVITIES[sel]),
                    family=_family_for(k),
                    num_tables=k,
                    selectivity=sel,
                )
            )
    if include_q5:
        for sel in selectivities:
            queries.append(
                WorkloadQuery(
                    query_id=f"Q5GB_{sel}",
                    sql=Q5_SQL.format(pred=SELECTIVITIES[sel]),
                    family="q5_groupby",
                    num_tables=5,
                    selectivity=sel,
                )
            )
    return queries


def load_sql_dir(path: str, family: str = "tpcds") -> List[WorkloadQuery]:
    """Load every ``*.sql`` file in ``path`` (sorted) as one query each."""

    queries: List[WorkloadQuery] = []
    for sql_file in sorted(Path(path).glob("*.sql")):
        sql = sql_file.read_text(encoding="utf-8").strip().rstrip(";")
        num_tables = sql.upper().count(" JOIN ") + 1
        queries.append(
            WorkloadQuery(
                query_id=sql_file.stem.upper(),
                sql=sql,
                family=family,
                num_tables=num_tables,
            )
        )
    return queries


def select_queries(queries: Sequence[WorkloadQuery], ids: Optional[Sequence[str]]) -> List[WorkloadQuery]:
    if not ids:
        return list(queries)
    wanted = {i.upper() for i in ids}
    return [q for q in queries if q.query_id.upper() in wanted or q.family.upper() in wanted]
