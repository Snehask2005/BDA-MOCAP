"""
Unified telemetry logging schema (owner: Anaswara).

Every measured execution in the project is written as ONE JSON line with
the 15 fields agreed in the work plan (Section 9B), in this order:

    query_id, plan_id, strategy, latency_ms, bytes_scanned, bytes_shuffled,
    executor_seconds, run_number, cache_state, aqe_enabled, cbo_enabled,
    num_cores, memory_gb, seed, timestamp

Extra fields (status, actual_cost, experiment, scenario, ...) are allowed
after the required ones, so later phases can add context without breaking
the schema.

Validate a log from the command line (used by the schema-check CI job):

    python -m execution.telemetry check results/telemetry/*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

TELEMETRY_FIELDS: tuple[str, ...] = (
    "query_id",
    "plan_id",
    "strategy",
    "latency_ms",
    "bytes_scanned",
    "bytes_shuffled",
    "executor_seconds",
    "run_number",
    "cache_state",
    "aqe_enabled",
    "cbo_enabled",
    "num_cores",
    "memory_gb",
    "seed",
    "timestamp",
)

# Expected types for the required fields. ``None`` is allowed only where
# listed in _NULLABLE (a failed run has no latency or byte counts).
_TYPES: Dict[str, tuple] = {
    "query_id": (str,),
    "plan_id": (str,),
    "strategy": (str,),
    "latency_ms": (int, float),
    "bytes_scanned": (int, float),
    "bytes_shuffled": (int, float),
    "executor_seconds": (int, float),
    "run_number": (int,),
    "cache_state": (str,),
    "aqe_enabled": (bool,),
    "cbo_enabled": (bool,),
    "num_cores": (int,),
    "memory_gb": (int, float),
    "seed": (int,),
    "timestamp": (int, float),
}
_NULLABLE = {"latency_ms", "bytes_scanned", "bytes_shuffled", "executor_seconds", "seed"}
CACHE_STATES = {"cold", "warm"}
STATUSES = {"ok", "failed", "cancelled"}


@dataclass
class TelemetryRecord:
    """One measured execution. Field order == the agreed schema."""

    query_id: str
    plan_id: str
    strategy: str
    latency_ms: Optional[float]
    bytes_scanned: Optional[float]
    bytes_shuffled: Optional[float]
    executor_seconds: Optional[float]
    run_number: int
    cache_state: str
    aqe_enabled: bool
    cbo_enabled: bool
    num_cores: int
    memory_gb: float
    seed: Optional[int]
    timestamp: float
    # ---- extensions (optional, not part of the frozen 15) ----
    status: str = "ok"
    error: Optional[str] = None
    actual_cost: Optional[float] = None
    executor_cpu_seconds: Optional[float] = None
    executor_seconds_source: str = "wall_clock_x_cores"
    experiment: Optional[str] = None
    scenario: Optional[str] = None
    plan_fingerprint: Optional[str] = None
    executed_fingerprint: Optional[str] = None
    plan_verified: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_record(record: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable schema problems (empty == valid)."""

    problems: List[str] = []
    for name in TELEMETRY_FIELDS:
        if name not in record:
            problems.append(f"missing field '{name}'")
            continue
        value = record[name]
        if value is None:
            if name not in _NULLABLE:
                problems.append(f"field '{name}' must not be null")
            continue
        expected = _TYPES[name]
        # bool is a subclass of int: reject bools in numeric fields.
        if bool not in expected and isinstance(value, bool):
            problems.append(f"field '{name}' has bool, expected {expected}")
            continue
        if not isinstance(value, expected):
            problems.append(
                f"field '{name}' has {type(value).__name__}, expected "
                + "/".join(t.__name__ for t in expected)
            )
            continue
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            problems.append(f"field '{name}' is not finite")

    if not problems:
        if record["cache_state"] not in CACHE_STATES:
            problems.append(f"cache_state must be one of {sorted(CACHE_STATES)}")
        if record["run_number"] < 1:
            problems.append("run_number must be >= 1")
        if record["num_cores"] < 1:
            problems.append("num_cores must be >= 1")
        for name in ("latency_ms", "bytes_scanned", "bytes_shuffled", "executor_seconds"):
            if record[name] is not None and record[name] < 0:
                problems.append(f"field '{name}' must be >= 0")
        status = record.get("status", "ok")
        if status not in STATUSES:
            problems.append(f"status must be one of {sorted(STATUSES)}")
        if status == "ok" and record["latency_ms"] is None:
            problems.append("an 'ok' run must have latency_ms")
    return problems


class TelemetryLogger:
    """Thread-safe JSON-lines appender for :class:`TelemetryRecord`."""

    def __init__(self, path: os.PathLike, validate: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.validate = validate
        self._lock = threading.Lock()

    def log(self, record: TelemetryRecord) -> Dict[str, Any]:
        row = record.to_dict()
        if self.validate:
            problems = validate_record(row)
            if problems:
                raise ValueError(
                    f"Telemetry record for {record.plan_id} violates schema: {problems}"
                )
        line = json.dumps(row, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return row


def read_log(path: os.PathLike) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def validate_file(path: os.PathLike) -> List[str]:
    """Validate every line of a telemetry log; returns located problems."""

    problems: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    problems.append(f"{path}:{number}: invalid JSON ({exc})")
                    continue
                for issue in validate_record(record):
                    problems.append(f"{path}:{number}: {issue}")
    except OSError as exc:
        problems.append(f"{path}: cannot read ({exc})")
    return problems


def to_dataframe(paths: Iterable[os.PathLike]):
    """Load one or more telemetry logs into a pandas DataFrame."""

    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(read_log(path))
    return pd.DataFrame(rows)


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MOCAP telemetry tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="validate telemetry JSONL files")
    check.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)

    files: List[str] = []
    for pattern in args.paths:
        matches = glob.glob(pattern, recursive=True)
        files.extend(matches if matches else [pattern])

    total = 0
    all_problems: List[str] = []
    for path in files:
        if not os.path.exists(path):
            continue
        total += 1
        all_problems.extend(validate_file(path))

    for problem in all_problems[:200]:
        print(problem)
    if all_problems:
        print(f"FAIL: {len(all_problems)} problem(s) in {total} file(s)")
        return 1
    print(f"OK: {total} telemetry file(s) match the unified schema")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
