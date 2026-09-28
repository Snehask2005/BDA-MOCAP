"""
Runtime monitoring and projected-cost calculation.

RuntimeMonitor polls Spark's statusTracker while a query is running and
projects total execution cost from observed progress.

Pricing is shared with the rest of MOCAP through PricingConfig.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from mocap.cost.pricing import PricingConfig, load_pricing_config


try:
    from pyspark.sql import SparkSession
except ImportError:  # pragma: no cover
    SparkSession = None


@dataclass
class ProgressSample:
    t: float
    completed_tasks: int
    total_tasks: int
    elapsed_s: float


class RuntimeMonitor:
    """
    Poll Spark's statusTracker on a background thread and compute a
    projected total execution cost.

    The projected cost uses the same PricingConfig as SparkExecutor.
    """

    def __init__(
        self,
        spark: Optional["SparkSession"],
        budget: float,
        poll_interval_s: float = 1.0,
        pricing: Optional[PricingConfig] = None,
        on_projected_overrun: Optional[
            Callable[[ProgressSample, float], None]
        ] = None,
    ):
        self.spark = spark
        self.budget = budget
        self.poll_interval_s = poll_interval_s
        self.pricing = pricing or load_pricing_config()
        self.on_projected_overrun = on_projected_overrun

        self._samples: List[ProgressSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    def start(self) -> None:
        self._start_time = time.time()
        self._stop_event.clear()
        self._samples = []

        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> List[ProgressSample]:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(
                timeout=self.poll_interval_s * 2
            )

        return list(self._samples)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._take_sample()

            if sample is not None:
                self._samples.append(sample)

                projected = self.project_total_cost(sample)

                if (
                    projected is not None
                    and projected > self.budget
                    and self.on_projected_overrun
                ):
                    self.on_projected_overrun(
                        sample,
                        projected,
                    )

            self._stop_event.wait(self.poll_interval_s)

    def _take_sample(self) -> Optional[ProgressSample]:
        if self.spark is None or self._start_time is None:
            return None

        tracker = self.spark.sparkContext.statusTracker()

        job_ids = tracker.getActiveJobIds()

        completed = 0
        total = 0

        for jid in job_ids:
            job_info = tracker.getJobInfo(jid)

            if job_info is None:
                continue

            for sid in job_info.stageIds:
                stage_info = tracker.getStageInfo(sid)

                if stage_info is None:
                    continue

                completed += stage_info.numCompletedTasks
                total += stage_info.numTasks

        return ProgressSample(
            t=time.time(),
            completed_tasks=completed,
            total_tasks=max(total, 1),
            elapsed_s=time.time() - self._start_time,
        )

    # ------------------------------------------------------------------
    # Cost projection
    # ------------------------------------------------------------------

    def project_total_cost(
        self,
        sample: ProgressSample,
    ) -> Optional[float]:
        """
        Project total monetary cost from task progress.

        The current model assumes that the average cost of completed
        tasks is representative of the remaining tasks.

        CPU cost is derived using the configured vCPU-hour price.
        Since the monitor currently observes wall-clock progress rather
        than executor CPU seconds, elapsed time is treated as a
        one-core CPU-second approximation.
        """
        if sample.completed_tasks <= 0:
            return None

        if sample.total_tasks <= 0:
            return None

        progress_fraction = (
            sample.completed_tasks / sample.total_tasks
        )

        if progress_fraction <= 0:
            return None

        projected_total_time = (
            sample.elapsed_s / progress_fraction
        )

        cpu_seconds = projected_total_time

        projected_cost = (
            cpu_seconds / 3600.0
        ) * self.pricing.cost_per_vcore_hour

        return round(projected_cost, 6)