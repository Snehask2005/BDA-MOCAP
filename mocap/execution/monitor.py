"""
Student 4 - Day 3-4 deliverable: runtime monitoring + projected-cost calculation.

While a query runs, RuntimeMonitor polls Spark's job/stage progress on a
background thread and extrapolates a "projected cost" for the full query.
adaptive.py uses this signal to decide whether to intervene (Section 8 /
Section 12, "End-to-End MOCAP Flow").
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

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
    Polls Spark's statusTracker on a background thread and computes a
    projected total cost by extrapolating from progress-so-far. Call
    `.start()` right before triggering the Spark action, `.stop()` right
    after it returns.
    """

    def __init__(
        self,
        spark: Optional["SparkSession"],
        budget: float,
        poll_interval_s: float = 1.0,
        cost_per_second: float = 0.0002,  # keep in sync with executor.py until pricing.py lands
        on_projected_overrun: Optional[Callable[[ProgressSample, float], None]] = None,
    ):
        self.spark = spark
        self.budget = budget
        self.poll_interval_s = poll_interval_s
        self.cost_per_second = cost_per_second
        self.on_projected_overrun = on_projected_overrun

        self._samples: List[ProgressSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    def start(self) -> None:
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[ProgressSample]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval_s * 2)
        return self._samples

    # ---- internals ----------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._take_sample()
            if sample is not None:
                self._samples.append(sample)
                projected = self.project_total_cost(sample)
                if projected is not None and projected > self.budget and self.on_projected_overrun:
                    self.on_projected_overrun(sample, projected)
            time.sleep(self.poll_interval_s)

    def _take_sample(self) -> Optional[ProgressSample]:
        if self.spark is None or self._start_time is None:
            return None
        tracker = self.spark.sparkContext.statusTracker()
        job_ids = tracker.getActiveJobIds()
        completed, total = 0, 0
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

    def project_total_cost(self, sample: ProgressSample) -> Optional[float]:
        """
        Linear extrapolation: assume remaining tasks cost the same, on
        average, as completed ones. Swap in Hridhika's estimator.py for a
        more accurate per-stage projection once it's ready (target: Sync 2,
        end of Day 6).
        """
        if sample.completed_tasks == 0:
            return None
        progress_fraction = sample.completed_tasks / sample.total_tasks
        if progress_fraction <= 0:
            return None
        projected_total_time = sample.elapsed_s / progress_fraction
        return round(projected_total_time * self.cost_per_second, 6)
