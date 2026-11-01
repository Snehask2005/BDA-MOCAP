"""
Cloud pricing profiles and bill reconciliation (owner: Anaswara, Phase 6).

Two cost views are reconciled:

* the MOCAP pricing model (mocap.cost.pricing, config/pricing.json) that the
  predictor and ``actual_cost`` telemetry use, and
* a provider profile (config/cloud_pricing.json) that converts measured
  executor-seconds into what EMR / Dataproc would charge for that work:

      cost = executor_seconds / 3600 x (instance $/h + service fee $/h) / vCPUs

Real bills are hourly per cluster, not per query, so reconciliation is done
per *batch*: run each batch on its own labelled cluster, export billing
grouped by that label, and compare billed vs telemetry totals per batch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLOUD_PRICING = ROOT / "config" / "cloud_pricing.json"


@dataclass(frozen=True)
class CloudProfile:
    provider: str
    instance: str
    vcpus: int
    compute_per_hour: float
    service_per_hour: float

    @property
    def per_vcpu_second(self) -> float:
        return (self.compute_per_hour + self.service_per_hour) / self.vcpus / 3600.0

    def run_cost(self, executor_seconds: Optional[float]) -> Optional[float]:
        if executor_seconds is None:
            return None
        return executor_seconds * self.per_vcpu_second

    def cluster_cost(self, wall_seconds: float, nodes: int) -> float:
        return wall_seconds / 3600.0 * nodes * (self.compute_per_hour + self.service_per_hour)


def load_profile(provider: str, instance: str, path: Optional[Path] = None) -> CloudProfile:
    data = json.loads(Path(path or DEFAULT_CLOUD_PRICING).read_text(encoding="utf-8"))
    if provider not in data:
        raise KeyError(f"provider '{provider}' not in {sorted(k for k in data if not k.startswith('_'))}")
    prov = data[provider]
    if instance not in prov["instances"]:
        raise KeyError(f"instance '{instance}' not in {sorted(prov['instances'])}")
    inst = prov["instances"][instance]
    vcpus = int(inst["vcpus"])
    if provider == "dataproc":
        service = float(prov.get("service_fee_per_vcpu_hour", 0.0)) * vcpus
    else:
        service = float(inst.get("emr_fee_per_hour", 0.0))
    return CloudProfile(provider, instance, vcpus, float(inst["compute_per_hour"]), service)


def spearman(x, y) -> Dict[str, float]:
    import numpy as np

    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "spearman": float("nan"), "p_value": float("nan"),
                "pearson": float("nan")}
    a = np.asarray([p[0] for p in pairs], dtype=float)
    b = np.asarray([p[1] for p in pairs], dtype=float)
    try:
        from scipy.stats import pearsonr, spearmanr

        rho, p = spearmanr(a, b)
        r, _ = pearsonr(a, b)
        return {"n": len(pairs), "spearman": float(rho), "p_value": float(p), "pearson": float(r)}
    except Exception:  # pragma: no cover
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return {"n": len(pairs), "spearman": float(np.corrcoef(ra, rb)[0, 1]), "p_value": float("nan"),
                "pearson": float(np.corrcoef(a, b)[0, 1])}
