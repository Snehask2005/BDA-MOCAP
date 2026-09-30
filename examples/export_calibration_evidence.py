"""
Export final PlanMetrics/telemetry evidence for the paper.

Consolidates everything a "prediction accuracy" / "calibration
improvement" paper result needs into one JSON file, generated from
stored logs only (never manually edited numbers, per the project's
scientific reporting rules):

- every recorded (prediction, telemetry) pair
- analytical vs. calibrated MAE/RMSE/MAPE
- the pricing assumptions in force when this was generated
- which calibration model version was used

Run from the repo root, after collect_calibration_samples.py and
mocap.calibration.trainer have produced results/calibration_dataset.csv
and results/calibration_model.json:

    python examples/export_calibration_evidence.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from mocap.calibration.dataset import load_dataset
from mocap.calibration.updater import evaluate
from mocap.cost.learned import CalibrationModel
from mocap.cost.pricing import load_pricing_config

CSV_PATH = "results/calibration_dataset.csv"
MODEL_PATH = "results/calibration_model.json"
OUT_PATH = "results/paper_evidence.json"


def main() -> None:
    rows = load_dataset(CSV_PATH)

    try:
        model = CalibrationModel.load(MODEL_PATH)
        model_available = True
    except FileNotFoundError:
        model = CalibrationModel.identity()
        model_available = False

    report = evaluate(CSV_PATH, model)
    pricing = load_pricing_config()

    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": CSV_PATH,
        "n_samples": len(rows),
        "calibration_model_available": model_available,
        "calibration_model_path": MODEL_PATH if model_available else None,
        "calibration_model_weights": model.weights.tolist(),
        "pricing_assumptions": {
            "currency": pricing.currency,
            "cost_per_vcore_hour": pricing.cost_per_vcore_hour,
            "cost_per_gb_scanned": pricing.cost_per_gb_scanned,
            "cost_per_gb_shuffled": pricing.cost_per_gb_shuffled,
        },
        "accuracy": report,
        "samples": rows,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"Wrote {len(rows)} samples and accuracy report to {OUT_PATH}")
    print(f"  analytical MAE={report['analytical']['MAE']:.6f}  calibrated MAE={report['calibrated']['MAE']:.6f}")


if __name__ == "__main__":
    main()
