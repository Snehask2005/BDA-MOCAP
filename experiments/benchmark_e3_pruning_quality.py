import csv
import math
import random
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates


BASE = Path("benchmarks/tpcds/representative/parquet")
SQL_PATH = Path("benchmarks/tpcds/representative/queries/q5_join_order.sql")
OUTPUT = Path("results/e3/unhinted_pruning_runs.csv")
SUMMARY = Path("results/e3/unhinted_pruning_summary.csv")

TABLES = [
    "store_sales",
    "item",
    "store",
    "date_dim",
    "customer_demographics",
]

BEAM_WIDTH = 10
REPETITIONS = 5
SEED = 20261025
ABS_TOL = 1e-8
REL_TOL = 1e-12


def equivalent(reference_rows, candidate_rows):
    if len(reference_rows) != len(candidate_rows):
        return False

    def index_rows(rows):
        result = {}
        for row in rows:
            key = tuple(row[:-1])
            if key in result:
                return None
            result[key] = row[-1]
        return result

    reference = index_rows(reference_rows)
    candidate = index_rows(candidate_rows)

    if reference is None or candidate is None:
        return False
    if reference.keys() != candidate.keys():
        return False

    for key, expected in reference.items():
        actual = candidate[key]

        if expected is None or actual is None:
            if expected is not actual:
                return False
            continue

        if not math.isclose(
            float(expected),
            float(actual),
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
        ):
            return False

    return True


def is_unhinted_candidate(candidate):
    if candidate.strategy == "default":
        return True

    if not candidate.strategy.startswith("join_order_"):
        return False

    return not candidate.strategy.endswith(
        ("_broadcast", "_merge", "_shuffle_hash")
    )


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-Unhinted-Pruning-Quality")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        for table in TABLES:
            table_path = (BASE / table).resolve()
            df = spark.read.parquet(table_path.as_uri())
            df.createOrReplaceTempView(table)
            print(f"{table}: {df.count()} rows")

        sql = SQL_PATH.read_text().strip()
        query = QueryRequest(
            query_id="TPCDS_PRUNING_5T",
            sql=sql,
            budget=100.0,
            deadline=100.0,
            accuracy_tolerance=0.0,
        )

        reference_rows = spark.sql(sql).collect()

        mode_fingerprints = {}
        plans = {}

        for mode in ("exhaustive", "beam"):
            candidates = generate_join_candidates(
                query=query,
                spark=spark,
                search_mode=mode,
                beam_width=BEAM_WIDTH,
            )
            selected = [
                c for c in candidates if is_unhinted_candidate(c)
            ]
            mode_fingerprints[mode] = {
                c.fingerprint for c in selected
            }

            for candidate in selected:
                plans.setdefault(candidate.fingerprint, candidate)

            print(
                f"{mode}: {len(selected)} unhinted candidates, "
                f"{len(mode_fingerprints[mode])} unique fingerprints"
            )

        if not mode_fingerprints["exhaustive"]:
            raise RuntimeError("No exhaustive unhinted candidates found.")
        if not mode_fingerprints["beam"]:
            raise RuntimeError("No beam unhinted candidates found.")

        rng = random.Random(SEED)
        fingerprints = list(plans)
        samples = {fp: [] for fp in fingerprints}

        # One warm-up per unique plan, in randomized order.
        rng.shuffle(fingerprints)
        print("\nWarm-up pass...")
        for fp in fingerprints:
            rows = spark.sql(plans[fp].sql).collect()
            if not equivalent(reference_rows, rows):
                raise RuntimeError(
                    f"Result equivalence failed for {plans[fp].strategy}"
                )

        # Interleave candidate measurements to reduce order bias.
        for repetition in range(1, REPETITIONS + 1):
            order = list(plans)
            rng.shuffle(order)
            print(f"Measured round {repetition}/{REPETITIONS}")

            for fp in order:
                start = time.perf_counter()
                rows = spark.sql(plans[fp].sql).collect()
                elapsed = time.perf_counter() - start

                if not equivalent(reference_rows, rows):
                    raise RuntimeError(
                        f"Result equivalence failed during measurement: "
                        f"{plans[fp].strategy}"
                    )

                samples[fp].append(elapsed)

        medians = {
            fp: statistics.median(values)
            for fp, values in samples.items()
        }

        exhaustive_best_fp = min(
            mode_fingerprints["exhaustive"],
            key=lambda fp: medians[fp],
        )
        beam_best_fp = min(
            mode_fingerprints["beam"],
            key=lambda fp: medians[fp],
        )

        exhaustive_best = medians[exhaustive_best_fp]
        beam_best = medians[beam_best_fp]
        pruning_loss = (
            (beam_best / exhaustive_best) - 1.0
        ) * 100.0

        output_rows = []
        for fp, candidate in plans.items():
            for repetition, elapsed in enumerate(samples[fp], start=1):
                output_rows.append({
                    "fingerprint": fp,
                    "strategy": candidate.strategy,
                    "in_exhaustive": fp in mode_fingerprints["exhaustive"],
                    "in_beam": fp in mode_fingerprints["beam"],
                    "repetition": repetition,
                    "execution_seconds": f"{elapsed:.6f}",
                    "median_execution_seconds": f"{medians[fp]:.6f}",
                    "result_equivalent": True,
                })

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(output_rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(output_rows)

        summary = {
            "query": "TPC-DS-derived five-table query",
            "strategy_scope": "default and unhinted join-order plans",
            "beam_width": BEAM_WIDTH,
            "exhaustive_candidate_count": len(
                mode_fingerprints["exhaustive"]
            ),
            "beam_candidate_count": len(mode_fingerprints["beam"]),
            "repetitions_per_candidate": REPETITIONS,
            "exhaustive_best_strategy": plans[
                exhaustive_best_fp
            ].strategy,
            "exhaustive_best_fingerprint": exhaustive_best_fp,
            "exhaustive_best_median_seconds": f"{exhaustive_best:.6f}",
            "beam_best_strategy": plans[beam_best_fp].strategy,
            "beam_best_fingerprint": beam_best_fp,
            "beam_best_median_seconds": f"{beam_best:.6f}",
            "exhaustive_best_retained": (
                exhaustive_best_fp in mode_fingerprints["beam"]
            ),
            "pruning_loss_percent": f"{pruning_loss:.4f}",
            "all_results_equivalent": True,
        }

        with SUMMARY.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(summary),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(summary)

        print("\n===== REPEATED PRUNING-LOSS SUMMARY =====")
        for key, value in summary.items():
            print(f"{key}: {value}")

        print(f"\nSaved detailed timings: {OUTPUT}")
        print(f"Saved summary: {SUMMARY}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
