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
OUT = Path("results")
TABLES = [
    "store_sales", "item", "store",
    "date_dim", "customer_demographics",
]
WIDTHS = (2, 5, 10, 20)
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

    ref = index_rows(reference_rows)
    got = index_rows(candidate_rows)

    if ref is None or got is None or ref.keys() != got.keys():
        return False

    for key, expected in ref.items():
        actual = got[key]
        if expected is None or actual is None:
            if expected is not actual:
                return False
        elif not math.isclose(
            float(expected), float(actual),
            rel_tol=REL_TOL, abs_tol=ABS_TOL,
        ):
            return False

    return True


def is_broadcast(candidate):
    return (
        candidate.strategy == "broadcast"
        or candidate.strategy.endswith("_broadcast")
    )


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-TPCDS-Beam-Width-Sweep")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        for table in TABLES:
            df = spark.read.parquet(str(BASE / table))
            df.createOrReplaceTempView(table)
            print(f"{table}: {df.count()} rows")

        sql = SQL_PATH.read_text().strip()
        query = QueryRequest(
            query_id="TPCDS_WIDTH_SWEEP",
            sql=sql,
            budget=100.0,
            deadline=100.0,
            accuracy_tolerance=0.0,
        )
        reference = spark.sql(sql).collect()

        modes = ["exhaustive"] + [f"beam_{w}" for w in WIDTHS]
        fingerprints_by_mode = {}
        generation_stats = {}
        plans = {}

        for mode in modes:
            width = None if mode == "exhaustive" else int(mode.split("_")[1])
            start = time.perf_counter()
            candidates = generate_join_candidates(
                query=query,
                spark=spark,
                search_mode="exhaustive" if width is None else "beam",
                beam_width=20 if width is None else width,
            )
            planning_seconds = time.perf_counter() - start

            selected = [c for c in candidates if is_broadcast(c)]
            fingerprints_by_mode[mode] = {
                c.fingerprint for c in selected
            }
            for candidate in selected:
                plans.setdefault(candidate.fingerprint, candidate)

            generation_stats[mode] = {
                "candidate_count_total": len(candidates),
                "broadcast_candidate_count": len(selected),
                "planning_seconds": planning_seconds,
            }
            print(
                f"{mode}: total={len(candidates)}, "
                f"broadcast={len(selected)}, "
                f"planning={planning_seconds:.4f}s"
            )

        if not fingerprints_by_mode["exhaustive"]:
            raise RuntimeError("No exhaustive broadcast candidates found.")

        rng = random.Random(SEED)
        fingerprints = list(plans)
        samples = {fp: [] for fp in fingerprints}

        print("\nWarm-up pass...")
        rng.shuffle(fingerprints)
        for fp in fingerprints:
            rows = spark.sql(plans[fp].sql).collect()
            if not equivalent(reference, rows):
                raise RuntimeError(
                    f"Result-equivalence failure: {plans[fp].strategy}"
                )

        for repetition in range(1, REPETITIONS + 1):
            order = list(plans)
            rng.shuffle(order)
            print(f"Measured round {repetition}/{REPETITIONS}")

            for fp in order:
                start = time.perf_counter()
                rows = spark.sql(plans[fp].sql).collect()
                elapsed = time.perf_counter() - start

                if not equivalent(reference, rows):
                    raise RuntimeError(
                        f"Result-equivalence failure during run: "
                        f"{plans[fp].strategy}"
                    )
                samples[fp].append(elapsed)

        medians = {
            fp: statistics.median(values)
            for fp, values in samples.items()
        }
        exhaustive_fps = fingerprints_by_mode["exhaustive"]
        exhaustive_best_fp = min(exhaustive_fps, key=lambda fp: medians[fp])
        exhaustive_best = medians[exhaustive_best_fp]

        summary_rows = []
        for mode in modes:
            fps = fingerprints_by_mode[mode]
            best_fp = min(fps, key=lambda fp: medians[fp])
            best_median = medians[best_fp]

            stats = generation_stats[mode]
            summary_rows.append({
                "mode": mode,
                **stats,
                "unique_broadcast_fingerprints": len(fps),
                "best_strategy": plans[best_fp].strategy,
                "best_fingerprint": best_fp,
                "best_median_execution_seconds": f"{best_median:.6f}",
                "exhaustive_best_retained": exhaustive_best_fp in fps,
                "pruning_loss_percent": (
                    f"{(best_median / exhaustive_best - 1) * 100:.4f}"
                ),
                "all_results_equivalent": True,
                "repetitions_per_candidate": REPETITIONS,
            })

        run_rows = []
        for fp, candidate in plans.items():
            for repetition, elapsed in enumerate(samples[fp], start=1):
                run_rows.append({
                    "fingerprint": fp,
                    "strategy": candidate.strategy,
                    "in_exhaustive": fp in fingerprints_by_mode["exhaustive"],
                    **{
                        f"in_{mode}": fp in fingerprints_by_mode[mode]
                        for mode in modes if mode != "exhaustive"
                    },
                    "repetition": repetition,
                    "execution_seconds": f"{elapsed:.6f}",
                    "median_seconds": f"{medians[fp]:.6f}",
                    "result_equivalent": True,
                })

        OUT.mkdir(parents=True, exist_ok=True)
        for filename, rows in [
            ("tpcds_beam_width_sweep_summary.csv", summary_rows),
            ("tpcds_beam_width_sweep_runs.csv", run_rows),
        ]:
            path = OUT / filename
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(rows[0]),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved: {path}")

        print("\n===== WIDTH SWEEP SUMMARY =====")
        for row in summary_rows:
            print(
                f"{row['mode']:12} "
                f"candidates={row['candidate_count_total']:3} "
                f"planning={row['planning_seconds']:.4f}s "
                f"best_median={row['best_median_execution_seconds']}s "
                f"loss={row['pruning_loss_percent']}% "
                f"exhaustive_best_retained={row['exhaustive_best_retained']}"
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
