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

import json
import os
from datetime import datetime, timezone

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


def train_and_save(
    csv_path: str,
    model_path: str,
) -> CalibrationModel:
    """
    Train a calibration model from a CSV dataset and save it.
    """

    model = train_from_csv(csv_path)

    model.save(model_path)

    return model


def train_and_version(
    csv_path: str,
    models_dir: str = "results/models",
    latest_path: str = "results/calibration_model.json",
) -> tuple[CalibrationModel, str]:
    """
    Train a model and save it two ways:

    1. A timestamped, immutable copy under `models_dir`.
    2. An overwrite of `latest_path`.

    Returns (model, path_to_versioned_copy).
    """

    model = train_from_csv(csv_path)

    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    n_samples = len(load_dataset(csv_path))
    versioned_filename = f"calibration_model_{timestamp}_n{n_samples}.json"
    versioned_path = os.path.join(models_dir, versioned_filename)

    model.save(versioned_path)
    model.save(latest_path)

    manifest_path = os.path.join(models_dir, "manifest.jsonl")
    with open(manifest_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "versioned_path": versioned_path,
                    "training_csv": csv_path,
                    "n_samples": n_samples,
                    "weights": model.weights.tolist(),
                }
            )
            + "\n"
        )

    return model, versioned_path