"""
Ablation configurations for the MOCAP optimizer.

Roshini (Member 3) — Core MOCAP Optimizer

Each AblationConfig removes or replaces one component of the MOCAP
optimizer so that its contribution can be measured independently.

Ablation variants (per Section 7, paper results required):
    A0  FULL_MOCAP          — all components enabled (baseline MOCAP)
    A1  NO_CALIBRATION      — use analytical predictor only (no learned cal.)
    A2  NO_PARETO           — skip Pareto filter; rank all feasible plans
    A3  NO_LAGRANGIAN       — replace Lagrangian ranking with min-cost
    A4  NO_ADAPTIVE         — ignore adaptive/AQP tolerance (accuracy_tolerance=0)
    A5  LATENCY_ONLY        — rank by latency only (alpha=1, beta=0)
    A6  COST_ONLY           — rank by cost only (alpha=0, beta=1)
    A7  HIGH_LAMBDA         — enable violation penalty (lambda_mult=10)

These configurations drive the ablation experiment scripts and are
referenced in the paper's ablation section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class AblationVariant(str, Enum):
    """Named ablation variants."""

    FULL_MOCAP = "full_mocap"
    NO_CALIBRATION = "no_calibration"
    NO_PARETO = "no_pareto"
    NO_LAGRANGIAN = "no_lagrangian"
    NO_ADAPTIVE = "no_adaptive"
    LATENCY_ONLY = "latency_only"
    COST_ONLY = "cost_only"
    HIGH_LAMBDA = "high_lambda"


@dataclass
class AblationConfig:
    """
    Configuration for one MOCAP ablation run.

    Parameters
    ----------
    variant : AblationVariant
        Identifies which component is ablated.
    description : str
        Human-readable description of the ablated component.
    use_calibration : bool
        When False, pass ``calibration_model=None`` to the predictor.
    use_pareto : bool
        When False, skip Pareto filtering (pass all feasible plans to ranker).
    use_lagrangian : bool
        When False, skip Lagrangian ranking and select by minimum cost.
    alpha : float
        Lagrangian weight for normalized latency.
    beta : float
        Lagrangian weight for normalized cost.
    gamma : float
        Lagrangian weight for accuracy proxy term.
    delta : float
        Lagrangian weight for variance term.
    lambda_mult : float
        Lagrangian violation penalty multiplier.
    force_accuracy_tolerance : float, optional
        When set, overrides the query's accuracy_tolerance (for A4 ablation).
    """

    variant: AblationVariant
    description: str

    use_calibration: bool = True
    use_pareto: bool = True
    use_lagrangian: bool = True

    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.0
    delta: float = 0.0
    lambda_mult: float = 0.0

    force_accuracy_tolerance: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "variant": self.variant.value,
            "description": self.description,
            "use_calibration": self.use_calibration,
            "use_pareto": self.use_pareto,
            "use_lagrangian": self.use_lagrangian,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "lambda_mult": self.lambda_mult,
            "force_accuracy_tolerance": self.force_accuracy_tolerance,
        }


# ---------------------------------------------------------------------------
# Standard ablation suite
# ---------------------------------------------------------------------------

ALL_ABLATIONS: Dict[AblationVariant, AblationConfig] = {
    AblationVariant.FULL_MOCAP: AblationConfig(
        variant=AblationVariant.FULL_MOCAP,
        description=(
            "Full MOCAP: calibration + Pareto filtering + Lagrangian ranking."
        ),
        use_calibration=True,
        use_pareto=True,
        use_lagrangian=True,
        alpha=0.5,
        beta=0.5,
    ),
    AblationVariant.NO_CALIBRATION: AblationConfig(
        variant=AblationVariant.NO_CALIBRATION,
        description=(
            "Ablation: analytical predictor only — no learned calibration."
        ),
        use_calibration=False,
        use_pareto=True,
        use_lagrangian=True,
        alpha=0.5,
        beta=0.5,
    ),
    AblationVariant.NO_PARETO: AblationConfig(
        variant=AblationVariant.NO_PARETO,
        description=(
            "Ablation: no Pareto filter — all feasible plans enter the ranker."
        ),
        use_calibration=True,
        use_pareto=False,
        use_lagrangian=True,
        alpha=0.5,
        beta=0.5,
    ),
    AblationVariant.NO_LAGRANGIAN: AblationConfig(
        variant=AblationVariant.NO_LAGRANGIAN,
        description=(
            "Ablation: no Lagrangian ranking — select minimum-cost feasible plan."
        ),
        use_calibration=True,
        use_pareto=True,
        use_lagrangian=False,
        alpha=0.5,
        beta=0.5,
    ),
    AblationVariant.NO_ADAPTIVE: AblationConfig(
        variant=AblationVariant.NO_ADAPTIVE,
        description=(
            "Ablation: accuracy tolerance forced to 0 — "
            "no AQP/degraded mode preference."
        ),
        use_calibration=True,
        use_pareto=True,
        use_lagrangian=True,
        alpha=0.5,
        beta=0.5,
        force_accuracy_tolerance=0.0,
    ),
    AblationVariant.LATENCY_ONLY: AblationConfig(
        variant=AblationVariant.LATENCY_ONLY,
        description=(
            "Ablation: rank by latency only (alpha=1, beta=0)."
        ),
        use_calibration=True,
        use_pareto=True,
        use_lagrangian=True,
        alpha=1.0,
        beta=0.0,
    ),
    AblationVariant.COST_ONLY: AblationConfig(
        variant=AblationVariant.COST_ONLY,
        description=(
            "Ablation: rank by cost only (alpha=0, beta=1)."
        ),
        use_calibration=True,
        use_pareto=True,
        use_lagrangian=True,
        alpha=0.0,
        beta=1.0,
    ),
    AblationVariant.HIGH_LAMBDA: AblationConfig(
        variant=AblationVariant.HIGH_LAMBDA,
        description=(
            "Ablation: high violation penalty (lambda_mult=10.0) — "
            "tests sensitivity of Lagrangian penalty term."
        ),
        use_calibration=True,
        use_pareto=True,
        use_lagrangian=True,
        alpha=0.5,
        beta=0.5,
        lambda_mult=10.0,
    ),
}


def get_ablation(variant: AblationVariant) -> AblationConfig:
    """Return the AblationConfig for the given variant."""
    return ALL_ABLATIONS[variant]


def list_ablations() -> Dict[str, Dict]:
    """Return all ablation configurations as serializable dicts."""
    return {v.value: cfg.to_dict() for v, cfg in ALL_ABLATIONS.items()}
