"""
Train the lightweight linear calibration model
(cpu_component, io_component, shuffle_component) -> actual_cost
using ordinary least squares (numpy only, no scikit-learn).

Usage:
    python -m mocap.calibration.trainer [csv_path] [out_path]
"""
from __future__ import annotations

import numpy as np

from mocap.calibration.dataset import load_dataset
from mocap.cost.learned import CalibrationModel


def train_from_csv(csv_path: str) -> CalibrationModel:
    rows = load_dataset(csv_path)
    if len(rows) < 4:
        raise ValueError(
            f"Need at least 4 labeled examples to fit 4 coefficients, got {len(rows)}. "
            "Run more candidates through estimate_from_execution + dataset.append_records first."
        )

    X = np.array(
        [[1.0, r["cpu_component"], r["io_component"], r["shuffle_component"]] for r in rows]
    )
    y = np.array([r["actual_cost"] for r in rows])

    weights, *_ = np.linalg.lstsq(X, y, rcond=None)
    return CalibrationModel(weights=weights)


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/calibration_dataset.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results/calibration_model.json"

    model = train_from_csv(csv_path)
    model.save(out_path)
    print(f"Trained calibration model saved to {out_path}")
    print(f"Weights [bias, cpu, io, shuffle]: {model.weights}")
