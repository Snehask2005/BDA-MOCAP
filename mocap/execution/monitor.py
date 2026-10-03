"""
Runtime monitoring and projected-cost calculation.

RuntimeMonitor polls Spark's statusTracker while a query is running and
projects total execution cost from observed progress.

Pricing is shared with the rest of MOCAP through PricingConfig, and the
CPU-seconds approximation matches SparkExecutor (wall-clock * cores), so
the projection and the final observed cost are on the same scale.

Known limitation: task progress is measured over the stages Spark has
submitted SO FAR. Later stages of a multi-stage query are not yet visible,
so early projections tend to be biased low. `min_elapsed_s` / `min_progress`
reduce noisy early decisions but do not remove this bias.
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

    job_group: if given, only jobs tagged with this group are observed
    (the executor sets the same group), so unrelated Spark activity does
    not distort the projection. Without it, all active jobs are used.

    The overrun callback fires at most ONCE per start().
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
        num_cores: int = 1,
        job_group: Optional[str] = None,
        min_elapsed_s: float = 0.0,
        min_progress: float = 0.0,
    ):
        if num_cores <= 0:
            raise ValueError("num_cores must be positive")

        self.spark = spark
        self.budget = budget
        self.poll_interval_s = poll_interval_s
        self.pricing = pricing or load_pricing_config()
        self.on_projected_overrun = on_projected_overrun
        self.num_cores = num_cores
        self.job_group = job_group
        self.min_elapsed_s = min_elapsed_s
        self.min_progress = min_progress

        self._samples: List[ProgressSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._fired = False

    def start(self) -> None:
        self._start_time = time.time()
        self._stop_event.clear()
        self._samples = []
        self._fired = False

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

    @property
    def overrun_fired(self) -> bool:
        return self._fired

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                sample = self._take_sample()
            except Exception as exc:  # a tracker hiccup must not kill monitoring
                print(f"WARNING: RuntimeMonitor sample failed: {exc}")
                sample = None

            if sample is not None:
                self._samples.append(sample)

                if not self._fired and self._eligible(sample):
                    projected = self.project_total_cost(sample)

                    if (
                        projected is not None
                        and projected > self.budget
                        and self.on_projected_overrun
                    ):
                        self._fired = True
                        try:
                            self.on_projected_overrun(sample, projected)
                        except Exception as exc:
                            print(f"WARNING: overrun callback failed: {exc}")

            self._stop_event.wait(self.poll_interval_s)

    def _eligible(self, sample: ProgressSample) -> bool:
        """Guard against acting on very early, noisy projections."""
        if sample.elapsed_s < self.min_elapsed_s:
            return False
        progress = sample.completed_tasks / max(sample.total_tasks, 1)
        return progress >= self.min_progress

    def _take_sample(self) -> Optional[ProgressSample]:
        if self.spark is None or self._start_time is None:
            return None

        tracker = self.spark.sparkContext.statusTracker()

        if self.job_group:
            job_ids = tracker.getJobIdsForGroup(self.job_group)
        else:
            job_ids = tracker.getActiveJobIds()

        completed = 0
        total = 0
        seen_stages = set()  # a stage shared by several jobs is counted once

        for jid in job_ids:
            job_info = tracker.getJobInfo(jid)

            if job_info is None:
                continue

            for sid in job_info.stageIds:
                if sid in seen_stages:
                    continue
                seen_stages.add(sid)

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

        Assumes the average cost of completed tasks is representative of
        the remaining tasks. Aggregate CPU-seconds are approximated as
        projected wall-clock time * num_cores, matching SparkExecutor.
        I/O and shuffle cost are not projected (bytes are unknown until
        the run finishes), so this is a CPU-only lower bound.
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

        cpu_seconds = projected_total_time * self.num_cores

        projected_cost = (
            cpu_seconds / 3600.0
        ) * self.pricing.cost_per_vcore_hour

        return round(projected_cost, 6)
