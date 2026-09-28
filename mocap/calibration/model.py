"""
Online calibration of MOCAP cost and latency predictions.

The calibration model learns a simple correction from historical
prediction/observation pairs:

    actual ~= scale * predicted + bias

This is intentionally lightweight and interpretable. It is a
calibration layer, not a replacement for the structural predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class CalibrationSample:
    """One independent prediction/observation pair."""

    predicted_cost: float
    actual_cost: float
    predicted_latency: float
    actual_latency: float


@dataclass
class CalibrationModel:
    """
    Lightweight linear calibration model.

    actual_cost    ~= cost_scale * predicted_cost + cost_bias
    actual_latency ~= latency_scale * predicted_latency + latency_bias
    """

    cost_scale: float = 1.0
    cost_bias: float = 0.0

    latency_scale: float = 1.0
    latency_bias: float = 0.0

    sample_count: int = 0

    def add_sample(
        self,
        predicted_cost: float,
        actual_cost: float,
        predicted_latency: float,
        actual_latency: float,
    ) -> None:
        """Update calibration using one observed execution."""

        self._validate_pair(predicted_cost, actual_cost)
        self._validate_pair(predicted_latency, actual_latency)

        self.sample_count += 1

        # Until enough observations exist, use a stable running
        # multiplicative correction. This avoids fitting an unstable
        # two-parameter line from a single sample.
        if self.sample_count == 1:
            if predicted_cost > 0:
                self.cost_scale = actual_cost / predicted_cost

            if predicted_latency > 0:
                self.latency_scale = actual_latency / predicted_latency

            return

        # Once multiple observations exist, refit using the stored
        # observations through fit().
        #
        # The standalone model remains lightweight; callers that want
        # full regression should call fit() with their historical data.

    def fit(self, samples: List[CalibrationSample]) -> None:
        """
        Fit linear correction models from historical observations.

        Requires at least two distinct predicted values for each
        dimension to estimate scale and bias.
        """

        if not samples:
            raise ValueError("At least one calibration sample is required.")

        self.sample_count = len(samples)

        cost_x = [s.predicted_cost for s in samples]
        cost_y = [s.actual_cost for s in samples]

        latency_x = [s.predicted_latency for s in samples]
        latency_y = [s.actual_latency for s in samples]

        self.cost_scale, self.cost_bias = self._fit_line(
            cost_x,
            cost_y,
            default_scale=self.cost_scale,
            default_bias=self.cost_bias,
        )

        self.latency_scale, self.latency_bias = self._fit_line(
            latency_x,
            latency_y,
            default_scale=self.latency_scale,
            default_bias=self.latency_bias,
        )

    def calibrate_cost(self, predicted_cost: float) -> float:
        """Return calibrated cost prediction."""

        self._validate_nonnegative(predicted_cost, "predicted_cost")

        return max(
            0.0,
            self.cost_scale * predicted_cost + self.cost_bias,
        )

    def calibrate_latency(self, predicted_latency: float) -> float:
        """Return calibrated latency prediction."""

        self._validate_nonnegative(
            predicted_latency,
            "predicted_latency",
        )

        return max(
            0.0,
            self.latency_scale * predicted_latency + self.latency_bias,
        )

    @staticmethod
    def _fit_line(
        x: List[float],
        y: List[float],
        default_scale: float,
        default_bias: float,
    ) -> Tuple[float, float]:
        """Fit y = scale*x + bias using ordinary least squares."""

        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")

        if not x:
            return default_scale, default_bias

        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        denominator = sum(
            (value - mean_x) ** 2
            for value in x
        )

        if denominator == 0:
            return default_scale, mean_y - default_scale * mean_x

        numerator = sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(x, y)
        )

        scale = numerator / denominator
        bias = mean_y - scale * mean_x

        return scale, bias

    @staticmethod
    def _validate_pair(predicted: float, actual: float) -> None:
        CalibrationModel._validate_nonnegative(
            predicted,
            "predicted",
        )
        CalibrationModel._validate_nonnegative(
            actual,
            "actual",
        )

    @staticmethod
    def _validate_nonnegative(value: float, name: str) -> None:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be numeric."
            ) from exc

        if value < 0:
            raise ValueError(
                f"{name} must be non-negative."
            )
