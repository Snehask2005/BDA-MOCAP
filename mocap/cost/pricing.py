"""
Pricing configuration and cost conversion utilities.

CPU usage is represented as aggregate CPU-seconds.

Therefore:

    CPU cost = CPU-seconds / 3600 * price per vCPU-hour

The CPU-seconds value must NOT be multiplied by the number of cores
again.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "pricing.json",
)


@dataclass(frozen=True)
class PricingConfig:
    currency: str
    cost_per_vcore_hour: float
    cost_per_gb_scanned: float
    cost_per_gb_shuffled: float
    cost_per_gb_hour_memory: float
    penalty_per_second: float

    @classmethod
    def from_dict(cls, data: dict) -> "PricingConfig":
        return cls(
            currency=data.get("currency", "USD"),
            cost_per_vcore_hour=data["cpu"]["cost_per_vcore_hour"],
            cost_per_gb_scanned=data["io"]["cost_per_gb_scanned"],
            cost_per_gb_shuffled=data["shuffle"]["cost_per_gb_shuffled"],
            cost_per_gb_hour_memory=data.get(
                "memory", {}
            ).get("cost_per_gb_hour", 0.0),
            penalty_per_second=data.get(
                "latency", {}
            ).get("penalty_per_second", 0.0),
        )


@lru_cache(maxsize=1)
def load_pricing_config(
    path: Optional[str] = None,
) -> PricingConfig:
    """Load pricing configuration."""

    config_path = path or _DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        default = {
            "currency": "USD",
            "cpu": {
                "cost_per_vcore_hour": 0.048,
            },
            "io": {
                "cost_per_gb_scanned": 0.005,
            },
            "shuffle": {
                "cost_per_gb_shuffled": 0.010,
            },
            "memory": {
                "cost_per_gb_hour": 0.006,
            },
            "latency": {
                "penalty_per_second": 0.0,
            },
        }

        return PricingConfig.from_dict(default)

    with open(config_path, "r") as f:
        data = json.load(f)

    return PricingConfig.from_dict(data)


def bytes_to_gb(num_bytes: float) -> float:
    """Convert bytes to GiB."""
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")

    return num_bytes / (1024 ** 3)


def cpu_cost(
    cpu_seconds: float,
    config: Optional[PricingConfig] = None,
) -> float:
    """
    Convert aggregate CPU-seconds to monetary cost.

    Example:

        3600 CPU-seconds
        × $0.048 / vCPU-hour
        = $0.048
    """

    if cpu_seconds < 0:
        raise ValueError(
            "cpu_seconds must be non-negative"
        )

    cfg = config or load_pricing_config()

    vcore_hours = cpu_seconds / 3600.0

    return (
        vcore_hours
        * cfg.cost_per_vcore_hour
    )


def io_cost(
    bytes_scanned: float,
    config: Optional[PricingConfig] = None,
) -> float:
    """Cost of scanning bytes from storage."""

    cfg = config or load_pricing_config()

    return (
        bytes_to_gb(bytes_scanned)
        * cfg.cost_per_gb_scanned
    )


def shuffle_cost(
    bytes_shuffled: float,
    config: Optional[PricingConfig] = None,
) -> float:
    """Cost of moving bytes through shuffle."""

    cfg = config or load_pricing_config()

    return (
        bytes_to_gb(bytes_shuffled)
        * cfg.cost_per_gb_shuffled
    )


def total_cost(
    cpu_seconds: float,
    bytes_scanned: float,
    bytes_shuffled: float,
    config: Optional[PricingConfig] = None,
) -> float:
    """Combine CPU, I/O and shuffle costs."""

    cfg = config or load_pricing_config()

    return (
        cpu_cost(cpu_seconds, cfg)
        + io_cost(bytes_scanned, cfg)
        + shuffle_cost(bytes_shuffled, cfg)
    )