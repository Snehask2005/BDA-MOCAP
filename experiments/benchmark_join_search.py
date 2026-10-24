from __future__ import annotations

import csv
import math
import statistics
import time
from pathlib import Path

from mocap.plans.generator import _generate_join_order_variants
from mocap.query.parser import QueryRequest


BEAM_WIDTHS = (10, 20, 50)
TABLE_COUNTS = (3, 4, 5, 6, 8, 10)
REPEATS = 30
WARMUPS = 3


def make_query(n_tables: int) -> QueryRequest:
    """Create a connected chain-join query for search benchmarking."""
    aliases = [f"t{i}" for i in range(n_tables)]
    table_names = [f"table_{i}" for i in range(n_tables)]

    sql = f"SELECT {aliases[0]}.id FROM {table_names[0]} {aliases[0]}"

    for i in range(1, n_tables):
        sql += (
            f" JOIN {table_names[i]} {aliases[i]}"
            f" ON {aliases[i-1]}.id = {aliases[i]}.prev_id"
        )

    return QueryRequest(
        query_id=f"SEARCH_{n_tables}",
        sql=sql,
        budget=100.0,
    )


def measure(
    query: QueryRequest,
    mode: str,
    width: int | None,
) -> dict:
    # Deterministic synthetic estimates are used only to exercise the
    # size-based pre-scorer; they are not claimed to be real table stats.
    n_tables = len(
        query.sql.split(" JOIN ")
    )
    aliases = [f"t{i}" for i in range(n_tables)]
    estimates = {
        alias: float(1000 // (i + 1) + 1)
        for i, alias in enumerate(aliases)
    }

    def generate():
        kwargs = {"search_mode": mode}
        if mode == "beam":
            kwargs["beam_width"] = width
            kwargs["table_size_estimates"] = estimates
        return _generate_join_order_variants(query, **kwargs)

    for _ in range(WARMUPS):
        generate()

    timings_ms = []
    final_variants = None

    for _ in range(REPEATS):
        start = time.perf_counter()
        variants = generate()
        timings_ms.append((time.perf_counter() - start) * 1000)
        final_variants = variants

    labels = [label for label, _sql in final_variants or []]

    if len(labels) != len(set(labels)):
        raise AssertionError("Duplicate join-order labels were generated.")

    if mode == "beam" and len(labels) > width:
        raise AssertionError("Beam search exceeded its configured width.")

    ordered_times = sorted(timings_ms)
    p95_index = math.ceil(0.95 * len(ordered_times)) - 1

    return {
        "table_count": n_tables,
        "search_mode": mode,
        "beam_width": width if width is not None else "",
        "valid_orders_generated": len(labels),
        "generation_median_ms": round(
            statistics.median(timings_ms), 4
        ),
        "generation_p95_ms": round(
            ordered_times[p95_index], 4
        ),
        "repetitions": REPEATS,
        "warmups": WARMUPS,
        "scoring": (
            "synthetic size-based heuristic"
            if mode == "beam"
            else "exhaustive reference"
        ),
    }


def main() -> None:
    rows = []

    for n_tables in TABLE_COUNTS:
        query = make_query(n_tables)

        if n_tables <= 5:
            row = measure(query, "exhaustive", None)
            rows.append(row)
            print(
                f"{n_tables:2d} tables | exhaustive | "
                f"orders={row['valid_orders_generated']:4d} | "
                f"median={row['generation_median_ms']:.4f} ms"
            )

        for width in BEAM_WIDTHS:
            row = measure(query, "beam", width)
            rows.append(row)
            print(
                f"{n_tables:2d} tables | beam {width:2d} | "
                f"orders={row['valid_orders_generated']:4d} | "
                f"median={row['generation_median_ms']:.4f} ms"
            )

    output = Path("results/search_scalability.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
