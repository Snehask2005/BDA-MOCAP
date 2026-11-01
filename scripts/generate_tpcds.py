"""
Generate official TPC-DS data and query text with DuckDB's tpcds extension
(owner: Anaswara).

DuckDB embeds the TPC-DS dsdgen generator, so no C toolkit has to be
compiled. Output layout matches what execution/data.py expects:

    benchmarks/tpcds/sf10/parquet/<table>/part-0.parquet
    benchmarks/tpcds/queries/q01.sql ... q99.sql

Usage:
    pip install duckdb
    python scripts/generate_tpcds.py --sf 10                 # ~3-4 GB parquet
    python scripts/generate_tpcds.py --sf 1 --queries-only   # just the 99 queries

Needs internet the first time (DuckDB downloads the extension). SF100 needs
a machine with ~150 GB free disk; generate it on a large VM and upload with
`gsutil -m cp -r` (Dataproc) or `aws s3 sync` (EMR). See docs/cluster_setup.md.

TPC-DS is a TPC benchmark: results produced with this data must be
described as "derived from TPC-DS" unless the full audited process is followed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sf", type=float, default=10, help="scale factor (1, 10, 100)")
    parser.add_argument("--out", default=None, help="output root (default benchmarks/tpcds/sf<SF>/parquet)")
    parser.add_argument("--queries-out", default=str(ROOT / "benchmarks" / "tpcds" / "queries"))
    parser.add_argument("--queries-only", action="store_true")
    parser.add_argument("--db", default=None, help="on-disk DuckDB file used while generating (spills to disk)")
    args = parser.parse_args(argv)

    import duckdb

    sf_label = f"{args.sf:g}"
    out = Path(args.out) if args.out else ROOT / "benchmarks" / "tpcds" / f"sf{sf_label}" / "parquet"
    db_path = args.db or str(ROOT / "benchmarks" / "tpcds" / f"_gen_sf{sf_label}.duckdb")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute("INSTALL tpcds")
    con.execute("LOAD tpcds")

    qdir = Path(args.queries_out)
    qdir.mkdir(parents=True, exist_ok=True)
    for number, text in con.execute("SELECT query_nr, query FROM tpcds_queries()").fetchall():
        (qdir / f"q{int(number):02d}.sql").write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Wrote 99 TPC-DS queries to {qdir}")
    if args.queries_only:
        return 0

    start = time.time()
    print(f"Generating TPC-DS SF{sf_label} (this can take a while) ...")
    con.execute(f"CALL dsdgen(sf={args.sf})")
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    out.mkdir(parents=True, exist_ok=True)
    for name in tables:
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        con.execute(
            f"COPY {name} TO '{(target / 'part-0.parquet').as_posix()}' (FORMAT parquet, COMPRESSION snappy)"
        )
        rows = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  {name:28} {rows:>12,} rows")
    con.close()
    Path(db_path).unlink(missing_ok=True)
    print(f"Done in {time.time() - start:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
