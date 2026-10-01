"""
Evaluate MOCAP cost calibration using a query-level held-out split.

The calibration model is trained only on training queries and evaluated
on candidate observations belonging to unseen query IDs.
"""

from __future__ import annotations

import csv
import os

from mocap.calibration.updater import evaluate_train_test


DATASET = "results/calibration_multi_query.csv"
OUTPUT = "results/calibration_evaluation.csv"


def main() -> None:
    result = evaluate_train_test(
        DATASET,
        test_fraction=0.25,
        random_seed=42,
    )

    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True,
    )

    rows = []

    for model_name in ("analytical", "calibrated"):
        metrics = result[model_name]

        rows.append(
            {
                "model": model_name,
                "mae": metrics["MAE"],
                "rmse": metrics["RMSE"],
                "mape_pct": metrics["MAPE_pct"],
                "train_samples": result["train_samples"],
                "test_samples": result["test_samples"],
                "train_queries": ",".join(result["train_queries"]),
                "test_queries": ",".join(result["test_queries"]),
            }
        )

    with open(
        OUTPUT,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "mae",
                "rmse",
                "mape_pct",
                "train_samples",
                "test_samples",
                "train_queries",
                "test_queries",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print("MOCAP HELD-OUT CALIBRATION EVALUATION")
    print("=" * 60)

    print(
        "Training queries:",
        result["train_queries"],
    )

    print(
        "Test queries:",
        result["test_queries"],
    )

    print(
        f"Training samples: {result['train_samples']}"
    )

    print(
        f"Test samples: {result['test_samples']}"
    )

    print()

    for model_name in ("analytical", "calibrated"):
        metrics = result[model_name]

        print(model_name.upper())
        print(f"  MAE:  {metrics['MAE']:.10f}")
        print(f"  RMSE: {metrics['RMSE']:.10f}")
        print(f"  MAPE: {metrics['MAPE_pct']:.2f}%")

    analytical = result["analytical"]
    calibrated = result["calibrated"]

    print()
    print("ERROR REDUCTION")

    for metric in ("MAE", "RMSE", "MAPE_pct"):
        baseline = analytical[metric]
        improved = calibrated[metric]

        if baseline != 0:
            reduction = (
                (baseline - improved)
                / baseline
                * 100.0
            )

            print(
                f"  {metric}: {reduction:.2f}%"
            )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
