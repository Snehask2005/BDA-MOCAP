"""
Unified plan runner + telemetry logger (owner: Anaswara).

Executes one SQL plan on Spark ``repeats`` times and writes one record per
measured run in the unified telemetry schema (execution/telemetry.py).

What is measured per run
------------------------
* latency_ms        wall-clock time of the full action (noop sink, so the
                    whole plan runs and no column pruning hides cost)
* bytes_scanned     sum of stage inputBytes        (Spark REST API)
* bytes_shuffled    sum of stage shuffleWriteBytes (Spark REST API)
* executor_seconds  sum of stage executorRunTime   (Spark REST API)
If the Spark UI / REST API is unavailable, bytes fall back to the SQL
metrics of the executed plan and executor_seconds to wall-clock x cores;
``executor_seconds_source`` records which one was used.

* actual_cost       mocap.cost.pricing.total_cost(executor_seconds,
                    bytes_scanned, bytes_shuffled): the same pricing model
                    the predictor uses, so predicted and actual cost are
                    in the same units.

Plan verification (reviewer guardrail)
--------------------------------------
When an expected fingerprint is given, the executed physical plan is
fingerprinted with the generator's own function and compared, so a hint
that Spark silently ignored is detected (``plan_verified``). Under AQE the
final plan is re-optimized at runtime, so verification is reported as
``None`` (not applicable) for AQE runs.

Command line (single query, quick check):

    python -m execution.run_plan --sql-file benchmarks/tpcds/e3/queries/q05_5_tables.sql \
        --tables-path benchmarks/tpcds/sf10/parquet --repeats 5
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from execution.config import Env, build_spark, load_env
from execution.telemetry import TelemetryLogger, TelemetryRecord

try:
    from mocap.cost.pricing import load_pricing_config, total_cost
except ImportError:  # pragma: no cover
    load_pricing_config = None
    total_cost = None

try:
    from mocap.plans.representation import create_plan_fingerprint
except ImportError:  # pragma: no cover
    create_plan_fingerprint = None

_SCAN_KEYS = ("size of files read",)
_SHUFFLE_KEYS = ("shuffle bytes written",)


@dataclass
class RunSpec:
    """Everything needed to execute and label one plan."""

    query_id: str
    plan_id: str
    strategy: str
    sql: str
    expected_fingerprint: Optional[str] = None
    conf_overrides: Optional[Dict[str, str]] = None


@contextlib.contextmanager
def temporary_conf(spark, overrides: Optional[Dict[str, str]]) -> Iterator[None]:
    """Set runtime SQL confs for one block, then restore the old values."""

    if not overrides:
        yield
        return
    previous: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        try:
            previous[key] = spark.conf.get(key)
        except Exception:
            previous[key] = None
        spark.conf.set(key, str(value))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                try:
                    spark.conf.unset(key)
                except Exception:
                    pass
            else:
                spark.conf.set(key, value)


class _SparkRest:
    """Minimal client for the driver's monitoring REST API."""

    def __init__(self, spark) -> None:
        sc = spark.sparkContext
        self.base = None
        try:
            url = sc.uiWebUrl
            if url:
                self.base = f"{url.rstrip('/')}/api/v1/applications/{sc.applicationId}"
        except Exception:
            self.base = None
        # Never route driver-local requests through an HTTP proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def available(self) -> bool:
        return self.base is not None

    def _get(self, path: str) -> Any:
        with self._opener.open(f"{self.base}{path}", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def job_group_totals(self, job_group: str, attempts: int = 20) -> Optional[Dict[str, float]]:
        """Sum stage metrics for every job tagged with ``job_group``."""

        if not self.available:
            return None
        for _ in range(attempts):
            try:
                jobs = [j for j in self._get("/jobs") if j.get("jobGroup") == job_group]
                if jobs and all(j.get("status") in ("SUCCEEDED", "FAILED") for j in jobs):
                    stage_ids = sorted({s for j in jobs for s in j.get("stageIds", [])})
                    totals = {
                        "executor_run_ms": 0.0,
                        "executor_cpu_ns": 0.0,
                        "input_bytes": 0.0,
                        "shuffle_write_bytes": 0.0,
                        "shuffle_read_bytes": 0.0,
                    }
                    complete = True
                    for stage_id in stage_ids:
                        try:
                            attempts_info = self._get(f"/stages/{stage_id}")
                        except Exception:
                            continue  # skipped stage (never ran)
                        for info in attempts_info:
                            if info.get("status") == "ACTIVE":
                                complete = False
                            totals["executor_run_ms"] += info.get("executorRunTime", 0) or 0
                            totals["executor_cpu_ns"] += info.get("executorCpuTime", 0) or 0
                            totals["input_bytes"] += info.get("inputBytes", 0) or 0
                            totals["shuffle_write_bytes"] += info.get("shuffleWriteBytes", 0) or 0
                            totals["shuffle_read_bytes"] += info.get("shuffleReadBytes", 0) or 0
                    if complete:
                        return totals
            except Exception:
                pass
            time.sleep(0.25)
        return None


class PlanRunner:
    """Executes plans with a fixed protocol and logs unified telemetry."""

    def __init__(
        self,
        spark,
        env: Env,
        logger: TelemetryLogger,
        experiment: Optional[str] = None,
        scenario: Optional[str] = None,
        pricing=None,
    ) -> None:
        self.spark = spark
        self.env = env
        self.logger = logger
        self.experiment = experiment
        self.scenario = scenario
        self.rest = _SparkRest(spark)
        if pricing is not None:
            self.pricing = pricing
        elif load_pricing_config is not None:
            self.pricing = load_pricing_config()
        else:  # pragma: no cover
            self.pricing = None

    # ------------------------------------------------------------------ API

    def run(
        self,
        spec: RunSpec,
        repeats: Optional[int] = None,
        warmup_runs: Optional[int] = None,
        cache_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Execute ``spec`` and return the logged telemetry rows."""

        exp = self.env.experiment
        repeats = int(repeats if repeats is not None else exp.get("repeats", 10))
        warmup_runs = int(warmup_runs if warmup_runs is not None else exp.get("warmup_runs", 1))
        cache_mode = cache_mode or exp.get("cache_mode", "warm")
        if cache_mode not in ("warm", "cold"):
            raise ValueError("cache_mode must be 'warm' or 'cold'")
        seed = self.env.seed if seed is None else seed

        rows: List[Dict[str, Any]] = []
        with temporary_conf(self.spark, spec.conf_overrides):
            for _ in range(warmup_runs if cache_mode == "warm" else 0):
                self._execute_once(spec)  # discarded
            for run_number in range(1, repeats + 1):
                if cache_mode == "cold":
                    self.spark.catalog.clearCache()
                    cache_state = "cold"
                else:
                    cache_state = "warm" if (warmup_runs > 0 or run_number > 1) else "cold"
                measurement = self._execute_once(spec)
                rows.append(self._log(spec, measurement, run_number, cache_state, seed))
        return rows

    # ------------------------------------------------------------ internals

    def _conf_bool(self, key: str) -> bool:
        try:
            return str(self.spark.conf.get(key)).lower() == "true"
        except Exception:
            return False

    def _execute_once(self, spec: RunSpec) -> Dict[str, Any]:
        sc = self.spark.sparkContext
        group = f"mocap-{spec.plan_id}-{uuid.uuid4().hex[:8]}"
        sc.setJobGroup(group, f"{spec.query_id}:{spec.plan_id}", interruptOnCancel=False)
        aqe = self._conf_bool("spark.sql.adaptive.enabled")
        cbo = self._conf_bool("spark.sql.cbo.enabled")
        start = time.perf_counter()
        wall_ts = time.time()
        try:
            df = self.spark.sql(spec.sql)
            df.write.format("noop").mode("overwrite").save()
            elapsed = time.perf_counter() - start
        except Exception as exc:  # failed run is logged, never retried
            return {
                "status": "failed",
                "error": str(exc)[:500],
                "elapsed": time.perf_counter() - start,
                "timestamp": wall_ts,
                "aqe": aqe,
                "cbo": cbo,
            }
        finally:
            sc.setJobGroup(f"mocap-idle-{uuid.uuid4().hex[:6]}", "idle", interruptOnCancel=False)

        executed_fp = None
        plan_bytes = (0.0, 0.0)
        try:
            jplan = df._jdf.queryExecution().executedPlan()
            if create_plan_fingerprint is not None:
                executed_fp = create_plan_fingerprint(jplan.toString())
            plan_bytes = self._plan_metric_bytes(jplan)
        except Exception:
            pass

        totals = self.rest.job_group_totals(group)
        if totals is not None:
            executor_seconds = totals["executor_run_ms"] / 1000.0
            executor_cpu = totals["executor_cpu_ns"] / 1e9
            bytes_scanned = totals["input_bytes"]
            bytes_shuffled = totals["shuffle_write_bytes"]
            source = "spark_rest_executorRunTime"
        else:
            executor_seconds = elapsed * self.env.num_cores
            executor_cpu = None
            bytes_scanned, bytes_shuffled = plan_bytes
            source = "wall_clock_x_cores"

        return {
            "status": "ok",
            "error": None,
            "elapsed": elapsed,
            "timestamp": wall_ts,
            "aqe": aqe,
            "cbo": cbo,
            "executor_seconds": executor_seconds,
            "executor_cpu": executor_cpu,
            "bytes_scanned": float(bytes_scanned),
            "bytes_shuffled": float(bytes_shuffled),
            "source": source,
            "executed_fp": executed_fp,
        }

    def _plan_metric_bytes(self, jplan) -> tuple:
        values: Dict[str, float] = {}

        def walk(node) -> None:
            it = node.metrics().iterator()
            while it.hasNext():
                entry = it.next()
                name = str(entry._1()).lower()
                try:
                    values[name] = values.get(name, 0.0) + float(entry._2().value())
                except Exception:
                    continue
            children = node.children().iterator()
            while children.hasNext():
                walk(children.next())

        walk(jplan)
        scanned = sum(v for k, v in values.items() if any(s in k for s in _SCAN_KEYS))
        shuffled = sum(v for k, v in values.items() if any(s in k for s in _SHUFFLE_KEYS))
        return scanned, shuffled

    def _cost(self, executor_seconds: float, scanned: float, shuffled: float) -> Optional[float]:
        if total_cost is None or self.pricing is None:
            return None
        return float(total_cost(executor_seconds, scanned, shuffled, config=self.pricing))

    def _log(self, spec, m, run_number, cache_state, seed) -> Dict[str, Any]:
        ok = m["status"] == "ok"
        verified = None
        if ok and spec.expected_fingerprint and not m["aqe"] and m.get("executed_fp"):
            verified = m["executed_fp"] == spec.expected_fingerprint
        record = TelemetryRecord(
            query_id=spec.query_id,
            plan_id=spec.plan_id,
            strategy=spec.strategy,
            latency_ms=(m["elapsed"] * 1000.0) if ok else None,
            bytes_scanned=m.get("bytes_scanned") if ok else None,
            bytes_shuffled=m.get("bytes_shuffled") if ok else None,
            executor_seconds=m.get("executor_seconds") if ok else None,
            run_number=int(run_number),
            cache_state=cache_state,
            aqe_enabled=bool(m["aqe"]),
            cbo_enabled=bool(m["cbo"]),
            num_cores=int(self.env.num_cores),
            memory_gb=float(self.env.memory_gb),
            seed=int(seed) if seed is not None else None,
            timestamp=float(m["timestamp"]),
            status=m["status"],
            error=m.get("error"),
            actual_cost=(
                self._cost(m["executor_seconds"], m["bytes_scanned"], m["bytes_shuffled"])
                if ok else None
            ),
            executor_cpu_seconds=m.get("executor_cpu"),
            executor_seconds_source=m.get("source", "none"),
            experiment=self.experiment,
            scenario=self.scenario,
            plan_fingerprint=spec.expected_fingerprint,
            executed_fingerprint=m.get("executed_fp"),
            plan_verified=verified,
        )
        return self.logger.log(record)


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one SQL plan with unified telemetry")
    parser.add_argument("--sql-file", required=True)
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--strategy", default="manual")
    parser.add_argument("--tables-path", default=None,
                        help="directory of <table>/ parquet folders to register as views")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--cache-mode", choices=["warm", "cold"], default=None)
    parser.add_argument("--log", default=None, help="output JSONL (default results/telemetry/run_plan.jsonl)")
    args = parser.parse_args(argv)

    env = load_env(profile=args.profile)
    spark = build_spark(env, "run_plan")
    try:
        if args.tables_path:
            from execution.data import register_parquet_dir
            register_parquet_dir(spark, args.tables_path)
        sql = Path(args.sql_file).read_text(encoding="utf-8")
        query_id = args.query_id or Path(args.sql_file).stem
        log_path = args.log or (env.output_dir("telemetry_dir") / "run_plan.jsonl")
        runner = PlanRunner(spark, env, TelemetryLogger(log_path), experiment="run_plan")
        rows = runner.run(
            RunSpec(query_id, args.plan_id or f"{query_id}_manual", args.strategy, sql),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            cache_mode=args.cache_mode,
        )
        for row in rows:
            print(
                f"run {row['run_number']:>2} {row['status']:>6} "
                f"latency={row['latency_ms'] or 0:.1f}ms "
                f"scanned={row['bytes_scanned'] or 0:.0f}B "
                f"shuffled={row['bytes_shuffled'] or 0:.0f}B "
                f"executor_s={row['executor_seconds'] or 0:.3f}"
            )
        print(f"Logged {len(rows)} run(s) to {log_path}")
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
