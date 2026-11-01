"""
Dataset registration, statistics and local data generation (owner: Anaswara).

Real experiments use official TPC-DS parquet data (SF10 / SF100) generated
with the TPC-DS toolkit -- see docs/cluster_setup.md. This module:

* registers a directory of ``<table>/`` parquet folders, either as temporary
  views (no statistics) or as catalog tables (statistics possible);
* runs ``ANALYZE TABLE ... COMPUTE STATISTICS FOR ALL COLUMNS`` so Spark CBO
  has real row/byte estimates (plan Phase 1 requirement);
* generates a SYNTHETIC TPC-DS-shaped star schema for laptop smoke tests and
  for the robustness suite (Zipf skew, data growth). Synthetic data is NOT
  TPC-DS and must never be reported as such in the paper.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Tables used by the E3 star-join workload (prepare_tpcds_e3_queries.py)
# and the representative Q5 query.
STAR_TABLES: tuple[str, ...] = (
    "store_sales",
    "date_dim",
    "item",
    "store",
    "customer_demographics",
    "time_dim",
    "customer",
    "customer_address",
    "household_demographics",
    "promotion",
)


def _uri(path: str) -> str:
    if "://" in path:
        return path
    return Path(path).resolve().as_uri()


def list_parquet_tables(base: str) -> List[str]:
    if "://" in base:
        raise ValueError("list_parquet_tables only works for local paths; pass tables= explicitly")
    root = Path(base)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


def register_parquet_dir(
    spark,
    base: str,
    tables: Optional[Iterable[str]] = None,
    mode: str = "view",
    database: str = "default",
) -> List[str]:
    """
    Register ``base/<table>`` parquet folders.

    mode="view"  -> temporary views (fast, no catalog statistics)
    mode="table" -> external catalog tables, so ANALYZE TABLE works
    """

    if mode not in ("view", "table"):
        raise ValueError("mode must be 'view' or 'table'")
    names = list(tables) if tables is not None else list_parquet_tables(base)
    registered: List[str] = []
    for name in names:
        location = f"{base.rstrip('/')}/{name}"
        if "://" not in location and not Path(location).exists():
            raise FileNotFoundError(f"Missing table directory: {location}")
        spark.catalog.dropTempView(name)
        if mode == "view":
            spark.read.parquet(_uri(location)).createOrReplaceTempView(name)
        else:
            spark.sql(f"DROP TABLE IF EXISTS {database}.{name}")
            spark.sql(
                f"CREATE TABLE {database}.{name} USING parquet LOCATION '{_uri(location)}'"
            )
        registered.append(name)
    return registered


def analyze_tables(spark, tables: Iterable[str], columns: bool = True) -> Dict[str, Optional[int]]:
    """Compute CBO statistics; returns the row count Spark recorded per table."""

    counts: Dict[str, Optional[int]] = {}
    for name in tables:
        spark.sql(f"ANALYZE TABLE {name} COMPUTE STATISTICS")
        if columns:
            spark.sql(f"ANALYZE TABLE {name} COMPUTE STATISTICS FOR ALL COLUMNS")
        counts[name] = table_row_estimate(spark, name)
    return counts


def table_row_estimate(spark, name: str) -> Optional[int]:
    """Row count from catalog statistics (None when no statistics exist)."""

    try:
        rows = spark.sql(f"DESCRIBE TABLE EXTENDED {name}").collect()
    except Exception:
        return None
    for row in rows:
        if str(row[0]).strip().lower() == "statistics":
            text = str(row[1])
            if "rows" in text:
                try:
                    return int(text.split(",")[1].strip().split(" ")[0])
                except Exception:
                    return None
    return None


def hadoop_copy_dir(spark, src: str, dst: str) -> None:
    """Copy a directory with Hadoop's FileSystem API (local, gs://, s3://, hdfs://)."""

    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    src_path = jvm.org.apache.hadoop.fs.Path(_uri(src))
    dst_path = jvm.org.apache.hadoop.fs.Path(_uri(dst))
    src_fs = src_path.getFileSystem(conf)
    dst_fs = dst_path.getFileSystem(conf)
    if dst_fs.exists(dst_path):
        dst_fs.delete(dst_path, True)
    jvm.org.apache.hadoop.fs.FileUtil.copy(src_fs, src_path, dst_fs, dst_path, False, conf)


def make_stale_tables(
    spark,
    base: str,
    grown: str,
    workdir: str,
    tables: Iterable[str],
    columns: bool = True,
) -> Dict[str, Dict[str, Optional[int]]]:
    """
    Register ``tables`` whose CBO statistics describe ``base`` but whose files
    are ``grown`` (stale statistics after data growth).

    The tables point at a working copy in ``workdir``: base data is copied in
    and ANALYZEd, then the files are replaced by the grown data and the file
    index refreshed. Catalog statistics only change on ANALYZE (or a
    Spark-managed write), so they stay at the base size.
    Returns {table: {"stats_rows": ..., "actual_rows": ...}}.
    """

    names = list(tables)
    for name in names:
        hadoop_copy_dir(spark, f"{base.rstrip('/')}/{name}", f"{workdir.rstrip('/')}/{name}")
    register_parquet_dir(spark, workdir, names, mode="table")
    analyze_tables(spark, names, columns=columns)
    for name in names:
        hadoop_copy_dir(spark, f"{grown.rstrip('/')}/{name}", f"{workdir.rstrip('/')}/{name}")
    refresh_tables(spark, names)
    report: Dict[str, Dict[str, Optional[int]]] = {}
    for name in names:
        report[name] = {
            "stats_rows": table_row_estimate(spark, name),
            "actual_rows": spark.table(name).count(),
        }
    return report


def refresh_tables(spark, tables: Iterable[str]) -> None:
    for name in tables:
        try:
            spark.sql(f"REFRESH TABLE {name}")
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Synthetic TPC-DS-shaped data (local smoke tests + robustness scenarios)
# ----------------------------------------------------------------------------

def _zipf_index(F, u_col, n: int, s: float):
    """Map a uniform(0,1) column to a Zipf-like index in [0, n-1].

    Continuous inverse-CDF approximation of a Zipf(s) law; s == 0 gives a
    uniform distribution.
    """

    if s <= 0:
        return F.floor(u_col * n).cast("long")
    if abs(s - 1.0) < 1e-9:
        expr = F.exp(u_col * F.log(F.lit(float(n) + 1.0))) - 1.0
    else:
        a = 1.0 - s
        expr = F.pow(u_col * (F.lit((n + 1.0) ** a) - 1.0) + 1.0, F.lit(1.0 / a)) - 1.0
    return F.least(F.floor(expr).cast("long"), F.lit(n - 1))


def synthetic_sizes(scale: float) -> Dict[str, int]:
    """Row counts per table for a synthetic scale (scale=1 ~ 200k facts)."""

    scale = max(scale, 0.01)
    return {
        "store_sales": int(200_000 * scale),
        "date_dim": 73_049,                          # TPC-DS fixed size
        "item": max(100, int(2_000 * scale)),
        "store": max(4, int(12 * scale ** 0.5)),
        "customer_demographics": 19_208,
        "time_dim": 86_400,                          # TPC-DS fixed size
        "customer": max(100, int(10_000 * scale)),
        "customer_address": max(50, int(5_000 * scale)),
        "household_demographics": 7_200,
        "promotion": max(10, int(30 * scale ** 0.5)),
    }


def generate_synthetic_star(
    spark,
    out_dir: str,
    scale: float = 1.0,
    zipf_s: float = 0.0,
    seed: int = 20261031,
    overwrite: bool = True,
) -> Dict[str, int]:
    """
    Write a synthetic star schema with the columns the E3 queries use.

    zipf_s > 0 skews store_sales foreign keys (item, customer, store, date)
    with a Zipf-like law -- used by the robustness "Zipfian skew" scenario.
    Returns the row count written per table.
    """

    from pyspark.sql import functions as F

    sizes = synthetic_sizes(scale)
    root = Path(out_dir)
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    def write(df, name: str) -> None:
        df.write.mode("overwrite").parquet(_uri(str(root / name)))

    n = sizes
    states = F.array(*[F.lit(s) for s in ("CA", "TX", "NY", "FL", "WA", "KL", "IL", "GA")])
    cats = F.array(*[F.lit(c) for c in ("Books", "Music", "Home", "Sports", "Shoes", "Jewelry")])

    write(
        spark.range(n["date_dim"]).select(
            F.col("id").alias("d_date_sk"),
            (F.lit(1900) + (F.col("id") / 365.25).cast("int")).alias("d_year"),
            ((F.col("id") % 365) / 31 + 1).cast("int").alias("d_moy"),
        ),
        "date_dim",
    )
    write(
        spark.range(n["item"]).select(
            F.col("id").alias("i_item_sk"),
            F.element_at(cats, (F.abs(F.hash("id", F.lit(1))) % 6 + 1).cast("int")).alias("i_category"),
            (F.abs(F.hash("id", F.lit(2))) % 10000 / 100.0).alias("i_current_price"),
        ),
        "item",
    )
    write(
        spark.range(n["store"]).select(
            F.col("id").alias("s_store_sk"),
            F.element_at(states, (F.col("id") % 8 + 1).cast("int")).alias("s_state"),
        ),
        "store",
    )
    write(
        spark.range(n["customer_demographics"]).select(
            F.col("id").alias("cd_demo_sk"),
            F.when(F.col("id") % 2 == 0, "M").otherwise("F").alias("cd_gender"),
        ),
        "customer_demographics",
    )
    write(
        spark.range(n["time_dim"]).select(
            F.col("id").alias("t_time_sk"),
            (F.col("id") / 3600).cast("int").alias("t_hour"),
        ),
        "time_dim",
    )
    write(
        spark.range(n["customer_address"]).select(
            F.col("id").alias("ca_address_sk"),
            F.element_at(states, (F.abs(F.hash("id", F.lit(3))) % 8 + 1).cast("int")).alias("ca_state"),
        ),
        "customer_address",
    )
    write(
        spark.range(n["customer"]).select(
            F.col("id").alias("c_customer_sk"),
            (F.abs(F.hash("id", F.lit(4))) % n["customer_address"]).alias("c_current_addr_sk"),
        ),
        "customer",
    )
    write(
        spark.range(n["household_demographics"]).select(
            F.col("id").alias("hd_demo_sk"),
            (F.col("id") % 10).cast("int").alias("hd_dep_count"),
        ),
        "household_demographics",
    )
    write(
        spark.range(n["promotion"]).select(
            F.col("id").alias("p_promo_sk"),
            F.when(F.col("id") % 3 == 0, "Y").otherwise("N").alias("p_channel_tv"),
        ),
        "promotion",
    )

    # Facts: dates concentrated in 1998-2003 so year predicates are selective.
    date_lo = int((1998 - 1900) * 365.25)
    date_span = int(6 * 365.25)
    facts = spark.range(n["store_sales"]).select(
        F.col("id").alias("ss_ticket_number"),
        F.rand(seed).alias("u1"),
        F.rand(seed + 1).alias("u2"),
        F.rand(seed + 2).alias("u3"),
        F.rand(seed + 3).alias("u4"),
        F.rand(seed + 5).alias("u5"),
    )
    facts = facts.select(
        "ss_ticket_number",
        (F.lit(date_lo) + _zipf_index(F, F.col("u1"), date_span, zipf_s)).alias("ss_sold_date_sk"),
        _zipf_index(F, F.col("u2"), n["item"], zipf_s).alias("ss_item_sk"),
        _zipf_index(F, F.col("u3"), n["store"], zipf_s).alias("ss_store_sk"),
        _zipf_index(F, F.col("u4"), n["customer"], zipf_s).alias("ss_customer_sk"),
        (F.abs(F.hash("ss_ticket_number", F.lit(5))) % n["customer_demographics"]).alias("ss_cdemo_sk"),
        (F.abs(F.hash("ss_ticket_number", F.lit(6))) % n["time_dim"]).alias("ss_sold_time_sk"),
        (F.abs(F.hash("ss_ticket_number", F.lit(7))) % n["household_demographics"]).alias("ss_hdemo_sk"),
        (F.abs(F.hash("ss_ticket_number", F.lit(8))) % n["promotion"]).alias("ss_promo_sk"),
        (F.col("u5") * 200.0).alias("ss_ext_sales_price"),
    )
    write(facts, "store_sales")
    return sizes


def ensure_synthetic(spark, out_dir: str, scale: float, zipf_s: float = 0.0, seed: int = 20261031) -> str:
    """Generate synthetic data once and reuse it (marker file per settings)."""

    marker = Path(out_dir) / f"_MOCAP_scale{scale}_zipf{zipf_s}_seed{seed}"
    if not marker.exists():
        generate_synthetic_star(spark, out_dir, scale=scale, zipf_s=zipf_s, seed=seed)
        marker.write_text("ok", encoding="utf-8")
    return out_dir


def directory_size_bytes(path: str) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(dirpath, name))
    return total
