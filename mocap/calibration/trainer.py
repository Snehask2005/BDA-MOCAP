"""
Train the MOCAP learned cost calibration model.

The model learns:

    actual_cost ≈ w0
                  + w1 * cpu_component
                  + w2 * io_component
                  + w3 * shuffle_component

using ordinary least squares.

Training data must contain independent execution observations.
"""

from __future__ import annotations

import numpy as np

from mocap.calibration.dataset import load_dataset
from mocap.cost.learned import CalibrationModel


def train_from_rows(rows) -> CalibrationModel:
    """
    Train the learned cost model from already-loaded calibration rows.

    Each row must contain:
        cpu_component
        io_component
        shuffle_component
        actual_cost
    """

    if len(rows) < 4:
        raise ValueError(
            "Need at least 4 labeled examples to fit 4 coefficients, "
            f"got {len(rows)}."
        )

    X = np.array(
        [
            [
                1.0,
                row["cpu_component"],
                row["io_component"],
                row["shuffle_component"],
            ]
            for row in rows
        ],
        dtype=float,
    )

    y = np.array(
        [
            row["actual_cost"]
            for row in rows
        ],
        dtype=float,
    )

    weights, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    return CalibrationModel(
        weights=weights,
    )


def train_from_csv(csv_path: str) -> CalibrationModel:
    """
    Train the learned cost model from a calibration dataset.
    """

    rows = load_dataset(csv_path)

    return train_from_rows(rows)


if __name__ == "__main__":
    import sys

    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "results/calibration_dataset.csv"
    )

    out_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "results/calibration_model.json"
    )

    model = train_from_csv(csv_path)

    model.save(out_path)

    print(
        f"Trained calibration model saved to {out_path}"
    )

    print(
        "Weights [bias, cpu, io, shuffle]:",
        model.weights,
    )