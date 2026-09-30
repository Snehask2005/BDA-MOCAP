import csv
import statistics
from pathlib import Path


INPUT_FILE = Path(
    "results/candidate_repeated_execution/repeated_execution_raw.csv"
)

OUTPUT_FILE = Path(
    "results/candidate_repeated_execution/candidate_execution_summary.csv"
)


def main():

    with INPUT_FILE.open(newline="") as file:
        rows = list(csv.DictReader(file))

    grouped = {}

    for row in rows:
        grouped.setdefault(row["plan_id"], []).append(row)

    summary_rows = []

    for plan_id, candidate_rows in grouped.items():

        latencies = [
            float(row["actual_latency_seconds"])
            for row in candidate_rows
        ]

        costs = [
            float(row["actual_cost"])
            for row in candidate_rows
        ]

        first = candidate_rows[0]

        summary_rows.append(
            {
                "plan_id": plan_id,
                "strategy": first["strategy"],
                "joins": first["joins"],
                "broadcasts": first["broadcasts"],
                "shuffle_hash": first["shuffle_hash"],
                "sort_merge": first["sort_merge"],
                "exchanges": first["exchanges"],
                "input_bytes": first["input_bytes"],
                "intermediate_bytes": first["intermediate_bytes"],
                "row_count_stat": first["row_count_stat"],
                "predicted_cost": first["predicted_cost"],
                "predicted_latency": first["predicted_latency"],
                "actual_latency_mean": statistics.mean(latencies),
                "actual_latency_median": statistics.median(latencies),
                "actual_latency_min": min(latencies),
                "actual_latency_max": max(latencies),
                "actual_cost_mean": statistics.mean(costs),
                "actual_cost_median": statistics.median(costs),
                "fingerprint": first["fingerprint"],
            }
        )

    # Sort by measured median latency only for convenient inspection.
    summary_rows.sort(
        key=lambda row: float(row["actual_latency_median"])
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 120)
    print("REPEATED EXECUTION SUMMARY")
    print("=" * 120)

    print(
        f"{'PLAN':<52}"
        f"{'PRED':>10}"
        f"{'MEDIAN':>12}"
        f"{'MIN':>10}"
        f"{'MAX':>10}"
    )

    print("-" * 120)

    for row in summary_rows:

        print(
            f"{row['plan_id']:<52}"
            f"{float(row['predicted_latency']):>10.4f}"
            f"{float(row['actual_latency_median']):>12.4f}"
            f"{float(row['actual_latency_min']):>10.4f}"
            f"{float(row['actual_latency_max']):>10.4f}"
        )

    print("-" * 120)

    print(f"\nNumber of raw measurements: {len(rows)}")
    print(f"Number of candidates: {len(summary_rows)}")
    print(f"Measurements per candidate: {len(rows) // len(summary_rows)}")

    print(f"\nSummary saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
