"""
Evaluate and refresh the calibration model.

- evaluate(): MAE / RMSE / MAPE for the raw analytical estimate and
  the calibrated estimate, against ground truth. This is the
  model-comparison / Day-9-10 evaluation deliverable.
- update_model(): retrain from the accumulated calibration dataset
  and overwrite the saved model file — call this periodically as
  more telemetry comes in.

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


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/calibration_dataset.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "results/calibration_model.json"

    model = update_model(csv_path, model_path)
    report = evaluate(csv_path, model)
    print("Evaluation (lower is better):")
    for name, errs in report.items():
        print(f"  {name}: MAE={errs['MAE']:.4f}  RMSE={errs['RMSE']:.4f}  MAPE={errs['MAPE_pct']:.2f}%")
