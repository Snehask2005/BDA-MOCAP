"""
Evaluate and refresh the calibration model.

- evaluate(): MAE / RMSE / MAPE for the raw analytical estimate and
  the calibrated estimate, against ground truth. This is the
  model-comparison / Day-9-10 evaluation deliverable.
- update_model(): retrain from the accumulated calibration dataset
  and overwrite the saved model file — call this periodically as
  more telemetry comes in.
- generate_markdown_report(): human/paper-readable table of the same
  numbers, no plotting dependency required.

Usage:
    python -m mocap.calibration.updater [csv_path] [model_path]
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from mocap.calibration.dataset import load_dataset
from mocap.calibration.trainer import train_from_csv
from mocap.cost.learned import CalibrationModel


def _errors(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs((pred[nonzero] - actual[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape}

def evaluate_train_test(
    csv_path: str,
    test_fraction: float = 0.3,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Train calibration on one set of queries and evaluate on
    completely held-out queries.

    Splitting is performed at query_id level to avoid placing
    different candidate plans from the same query in both
    training and test sets.
    """


    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            "test_fraction must be between 0 and 1."
        )

    rows = load_dataset(csv_path)

    if len(rows) < 8:
        raise ValueError(
            "Need at least 8 observations for train/test validation, "
            f"got {len(rows)}."
        )

    query_ids = sorted(
        {
            row["query_id"]
            for row in rows
        }
    )

    if len(query_ids) < 2:
        raise ValueError(
            "Need at least 2 distinct query_id values for "
            "query-level train/test validation."
        )

    rng = np.random.default_rng(random_seed)

    shuffled_ids = list(query_ids)
    rng.shuffle(shuffled_ids)

    test_query_count = max(
        1,
        int(round(len(shuffled_ids) * test_fraction)),
    )

    if test_query_count >= len(shuffled_ids):
        test_query_count = len(shuffled_ids) - 1

    test_ids = set(
        shuffled_ids[:test_query_count]
    )

    train_rows = [
        row
        for row in rows
        if row["query_id"] not in test_ids
    ]

    test_rows = [
        row
        for row in rows
        if row["query_id"] in test_ids
    ]

    if len(train_rows) < 4:
        raise ValueError(
            "Training split must contain at least 4 observations."
        )

    from mocap.calibration.trainer import train_from_rows

    model = train_from_rows(train_rows)

    actual = np.array(
        [
            row["actual_cost"]
            for row in test_rows
        ],
        dtype=float,
    )

    analytical_pred = np.array(
        [
            row["estimated_cost"]
            for row in test_rows
        ],
        dtype=float,
    )

    calibrated_pred = np.array(
        [
            model.predict(
                row["cpu_component"],
                row["io_component"],
                row["shuffle_component"],
            )
            for row in test_rows
        ],
        dtype=float,
    )

    return {
        "train_queries": sorted(
            {
                row["query_id"]
                for row in train_rows
            }
        ),
        "test_queries": sorted(
            {
                row["query_id"]
                for row in test_rows
            }
        ),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "analytical": _errors(
            analytical_pred,
            actual,
        ),
        "calibrated": _errors(
            calibrated_pred,
            actual,
        ),
    }


def evaluate(csv_path: str, model: CalibrationModel) -> Dict[str, Dict[str, float]]:
    rows = load_dataset(csv_path)
    actual = np.array([r["actual_cost"] for r in rows])
    analytical_pred = np.array([r["estimated_cost"] for r in rows])
    calibrated_pred = np.array(
        [model.predict(r["cpu_component"], r["io_component"], r["shuffle_component"]) for r in rows]
    )

    return {
        "analytical": _errors(analytical_pred, actual),
        "calibrated": _errors(calibrated_pred, actual),
    }


def update_model(csv_path: str, model_path: str) -> CalibrationModel:
    """Retrain on the current dataset and overwrite the saved model."""
    model = train_from_csv(csv_path)
    model.save(model_path)
    return model


def plot_estimated_vs_actual(csv_path: str, out_path: str) -> None:
    """
    Optional Day-9-10 deliverable: estimated-vs-actual scatter plot.
    Requires matplotlib, which is NOT currently in requirements.txt —
    add it (`pip install matplotlib`) before calling this.
    """
    import matplotlib.pyplot as plt  # local import: optional dependency

    rows = load_dataset(csv_path)
    actual = [r["actual_cost"] for r in rows]
    estimated = [r["estimated_cost"] for r in rows]

    plt.figure()
    plt.scatter(actual, estimated, alpha=0.6)
    lo, hi = min(actual + estimated), max(actual + estimated)
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    plt.xlabel("Actual cost")
    plt.ylabel("Estimated cost")
    plt.title("Analytical estimate vs actual cost")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def generate_markdown_report(csv_path: str, model: CalibrationModel, out_path: str) -> str:
    """
    Write a markdown table of per-sample predictions plus the
    aggregate MAE/RMSE/MAPE comparison -- the "calibration tables"
    checklist deliverable. Does not require matplotlib.
    """
    rows = load_dataset(csv_path)
    report = evaluate(csv_path, model)

    lines = [
        "# Calibration Report",
        "",
        f"Samples: {len(rows)}",
        "",
        "## Accuracy (lower is better)",
        "",
        "| Model | MAE | RMSE | MAPE (%) |",
        "|---|---|---|---|",
        f"| Analytical (uncalibrated) | {report['analytical']['MAE']:.6f} | "
        f"{report['analytical']['RMSE']:.6f} | {report['analytical']['MAPE_pct']:.2f} |",
        f"| Calibrated | {report['calibrated']['MAE']:.6f} | "
        f"{report['calibrated']['RMSE']:.6f} | {report['calibrated']['MAPE_pct']:.2f} |",
        "",
        "## Per-sample predictions",
        "",
        "| plan_id | query_id | estimated_cost | calibrated_cost | actual_cost |",
        "|---|---|---|---|---|",
    ]

    for row in rows:
        calibrated_cost = model.predict(row["cpu_component"], row["io_component"], row["shuffle_component"])
        lines.append(
            f"| {row['plan_id']} | {row['query_id']} | {row['estimated_cost']:.6f} | "
            f"{calibrated_cost:.6f} | {row['actual_cost']:.6f} |"
        )

    content = "\n".join(lines) + "\n"

    with open(out_path, "w") as f:
        f.write(content)

    return content


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/calibration_dataset.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "results/calibration_model.json"
    report_path = sys.argv[3] if len(sys.argv) > 3 else "results/calibration_report.md"

    model = update_model(csv_path, model_path)
    report = evaluate(csv_path, model)
    print("Evaluation (lower is better):")
    for name, errs in report.items():
        print(f"  {name}: MAE={errs['MAE']:.4f}  RMSE={errs['RMSE']:.4f}  MAPE={errs['MAPE_pct']:.2f}%")

    generate_markdown_report(csv_path, model, report_path)
    print(f"\nMarkdown report written to {report_path}")
