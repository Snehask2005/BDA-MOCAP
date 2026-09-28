"""
Calibration service for MOCAP.

Connects pre-execution PlanMetrics with post-execution
ExecutionTelemetry and maintains a calibration dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from mocap.cost.analytical import PlanMetrics
from mocap.interfaces import ExecutionTelemetry


@dataclass(frozen=True)
class CalibrationSample:
    """One prediction/observation pair."""

    plan_id: str
    query_id: str

    estimated_cost: float
    actual_cost: float

    estimated_latency: float
    actual_latency: float

    cpu_component: float
    io_component: float
    shuffle_component: float


class CalibrationService:
    """
    Collects independent execution observations for calibration.

    The service does not execute queries. It only connects predictions
    with telemetry that was collected after execution.
    """

    def __init__(self) -> None:
        self._samples: List[CalibrationSample] = []

    @property
    def samples(self) -> List[CalibrationSample]:
        """Return a copy of the collected samples."""
        return list(self._samples)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def record(
        self,
        prediction: PlanMetrics,
        telemetry: ExecutionTelemetry,
    ) -> CalibrationSample:
        """
        Record one independent prediction/observation pair.
        """

        if prediction.plan_id != telemetry.plan_id:
            raise ValueError(
                "Prediction and telemetry must refer to the same plan."
            )

        if prediction.query_id != telemetry.query_id:
            raise ValueError(
                "Prediction and telemetry must refer to the same query."
            )

        sample = CalibrationSample(
            plan_id=prediction.plan_id,
            query_id=prediction.query_id,
            estimated_cost=prediction.estimated_cost,
            actual_cost=telemetry.actual_cost,
            estimated_latency=prediction.estimated_latency,
            actual_latency=telemetry.actual_latency,
            cpu_component=prediction.cpu_component,
            io_component=prediction.io_component,
            shuffle_component=prediction.shuffle_component,
        )

        self._samples.append(sample)

        return sample
    
    def record_to_dataset(
        self,
        prediction: PlanMetrics,
        telemetry: ExecutionTelemetry,
        csv_path: str,
    ) -> CalibrationSample:
        """
        Record an execution observation and persist it to the
        calibration dataset.
        """

        sample = self.record(
            prediction=prediction,
            telemetry=telemetry,
        )

        from mocap.calibration.dataset import append_samples

        append_samples(
            [sample],
            csv_path,
        )

        return sample

    def build_cost_training_rows(self) -> List[dict]:
        """
        Convert observations into rows compatible with the existing
        learned calibration trainer.
        """

        return [
            {
                "plan_id": sample.plan_id,
                "query_id": sample.query_id,
                "cpu_component": sample.cpu_component,
                "io_component": sample.io_component,
                "shuffle_component": sample.shuffle_component,
                "estimated_cost": sample.estimated_cost,
                "actual_cost": sample.actual_cost,
            }
            for sample in self._samples
        ]