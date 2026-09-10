"""
Learned cost-correction layer.

Wraps the analytical PlanMetrics with a lightweight linear
calibration correction learned from held-out (estimated, actual)
pairs. No external ML dependency (scikit-learn isn't in
requirements.txt) — the model is a simple per-component linear
regression solved with numpy.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mocap.cost.analytical import PlanMetrics

# experiments/mocap/cost/learned.py -> experiments/results/calibration_model.json
_DEFAULT_MODEL_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "results",
    "calibration_model.json",
)


@dataclass
class CalibrationModel:
    """
    Per-component linear correction:
        calibrated_cost = w0 + w1*cpu_component + w2*io_component + w3*shuffle_component
    Trained by mocap/calibration/trainer.py.
    """

    weights: np.ndarray  # shape (4,): [bias, w_cpu, w_io, w_shuffle]

    def predict(self, cpu_component: float, io_component: float, shuffle_component: float) -> float:
        x = np.array([1.0, cpu_component, io_component, shuffle_component])
        return float(np.dot(self.weights, x))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"weights": self.weights.tolist()}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationModel":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(weights=np.array(data["weights"]))

    @classmethod
    def identity(cls) -> "CalibrationModel":
        """No-op calibration: calibrated cost == raw analytical cost."""
        return cls(weights=np.array([0.0, 1.0, 1.0, 1.0]))


def calibrate(metrics: PlanMetrics, model: Optional[CalibrationModel] = None) -> float:
    """
    Return a calibrated cost estimate for `metrics`. Falls back to
    the identity model (raw analytical cost) if no trained model
    file exists yet.
    """
    if model is None:
        try:
            model = CalibrationModel.load(_DEFAULT_MODEL_JSON)
        except FileNotFoundError:
            model = CalibrationModel.identity()

    return model.predict(metrics.cpu_component, metrics.io_component, metrics.shuffle_component)
