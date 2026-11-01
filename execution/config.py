"""
Shared environment loader (owner: Anaswara).

Every experiment builds its SparkSession through :func:`build_spark` so the
four developers' runs use identical settings (configs/spark_env.yaml).
"""

from __future__ import annotations

import copy
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = ROOT / "configs" / "spark_env.yaml"


@dataclass
class Env:
    """Resolved environment: shared settings merged with one profile."""

    profile: str
    master: Optional[str]
    app_name: str
    num_cores: int
    memory_gb: float
    spark_conf: Dict[str, str]
    seeds: Dict[str, int]
    data: Dict[str, str]
    experiment: Dict[str, Any]
    outputs: Dict[str, str]
    versions: Dict[str, str] = field(default_factory=dict)

    @property
    def seed(self) -> int:
        return int(self.seeds.get("global", 0))

    def output_dir(self, key: str) -> Path:
        path = Path(self.outputs[key])
        if not path.is_absolute():
            path = ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def data_path(self, key: str) -> str:
        value = self.data[key]
        if "://" in value or Path(value).is_absolute():
            return value
        return str(ROOT / value)


def load_env(
    path: Optional[os.PathLike] = None,
    profile: Optional[str] = None,
) -> Env:
    """Load configs/spark_env.yaml and resolve one profile."""

    config_path = Path(path) if path else DEFAULT_ENV_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    profile = profile or os.environ.get("MOCAP_PROFILE", "local")
    profiles = raw.get("profiles", {})
    if profile not in profiles:
        raise KeyError(
            f"Unknown profile '{profile}'. Available: {sorted(profiles)}"
        )
    prof = profiles[profile]

    conf = {str(k): str(v) for k, v in raw.get("spark_conf", {}).items()}
    conf.update({str(k): str(v) for k, v in prof.get("spark_conf", {}).items()})

    return Env(
        profile=profile,
        master=prof.get("master"),
        app_name=prof.get("app_name", "MOCAP"),
        num_cores=int(prof.get("num_cores", 1)),
        memory_gb=float(prof.get("memory_gb", 0)),
        spark_conf=conf,
        seeds=dict(raw.get("seeds", {})),
        data=dict(raw.get("data", {})),
        experiment=copy.deepcopy(raw.get("experiment", {})),
        outputs=dict(raw.get("outputs", {})),
        versions=dict(raw.get("versions", {})),
    )


def build_spark(env: Env, app_suffix: str = "", extra_conf: Optional[Dict[str, str]] = None):
    """Create (or reuse) a SparkSession configured from ``env``."""

    from pyspark.sql import SparkSession

    # Pin worker interpreter (avoids the Windows Store python alias issue).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    builder = SparkSession.builder.appName(
        f"{env.app_name}{('-' + app_suffix) if app_suffix else ''}"
    )
    if env.master:
        builder = builder.master(env.master)

    conf = dict(env.spark_conf)
    if extra_conf:
        conf.update({k: str(v) for k, v in extra_conf.items()})
    for key, value in conf.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Runtime-settable SQL confs must be re-applied when getOrCreate()
    # returned an already-running session.
    for key, value in conf.items():
        if key.startswith("spark.sql."):
            try:
                spark.conf.set(key, value)
            except Exception:
                pass
    return spark


def runtime_description(env: Env) -> Dict[str, Any]:
    """Environment facts recorded alongside every result file."""

    try:
        import pyspark
        spark_version = pyspark.__version__
    except Exception:  # pragma: no cover
        spark_version = None
    return {
        "profile": env.profile,
        "python": platform.python_version(),
        "pyspark": spark_version,
        "platform": platform.platform(),
        "num_cores": env.num_cores,
        "memory_gb": env.memory_gb,
        "seed": env.seed,
    }
