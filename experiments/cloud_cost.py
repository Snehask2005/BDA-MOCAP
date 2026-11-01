"""
Phase 6 -- Real cloud cost validation (owner: Anaswara).

Three checks, written to results/phase6/cloud_cost/:

1. predicted_vs_cloud   per plan: MOCAP's PRE-EXECUTION predicted cost vs the
                        provider cost of its measured executor-seconds
                        (Spearman rank correlation). Needs --measurements.
2. model_vs_cloud       per run: MOCAP pricing-model cost (telemetry
                        actual_cost) vs provider cost of the same run.
3. telemetry_vs_bill    per batch: telemetry-derived provider cost vs the
                        REAL billed amount from a billing export (--bill).
                        This is the "real cloud invoice" figure for the paper:
                        paper/figures/real_cost.pdf

Billing export format (--bill): a CSV with one row per batch, e.g. from the
GCP BigQuery billing export grouped by the cluster label ``mocap-batch``, or
AWS Cost Explorer grouped by the tag ``mocap-batch``:

    batch,cost
    e1_main,4.82
    robustness,7.10

``batch`` must match the telemetry field chosen with --batch-field
(default ``experiment``). See docs/cluster_setup.md for the labelling steps.

Example:
    python experiments/cloud_cost.py --provider dataproc --instance n2-standard-8 \
        --telemetry "results/telemetry/*.jsonl" \
        --measurements results/phase5/tpcds_sf10/measurements --bill bills/dataproc.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.cli import write_csv  # noqa: E402
from execution.cloud import load_profile, spearman  # noqa: E402
from execution.config import load_env  # noqa: E402
from execution.harness import load_measurements  # noqa: E402
from execution.telemetry import read_log  # noqa: E402


def scatter(x, y, labels, xlabel, ylabel, title, out_base: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(x, y, s=18)
    lo = min(min(x), min(y))
    hi = max(max(x), max(y))
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1)
    if labels and len(labels) <= 25:
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(str(lab), (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["dataproc", "emr"], default="dataproc")
    parser.add_argument("--instance", default="n2-standard-8")
    parser.add_argument("--telemetry", nargs="*", default=["results/telemetry/*.jsonl"])
    parser.add_argument("--measurements", default=None, help="measurement cache dir from E1")
    parser.add_argument("--bill", default=None, help="billing export CSV (batch,cost)")
    parser.add_argument("--bill-batch-col", default="batch")
    parser.add_argument("--bill-cost-col", default="cost")
    parser.add_argument("--batch-field", default="experiment")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    env = load_env()
    profile = load_profile(args.provider, args.instance)
    out_dir = Path(args.out) if args.out else env.output_dir("phase6_dir") / "cloud_cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = env.output_dir("figures_dir")
    report = {"provider": profile.provider, "instance": profile.instance,
              "usd_per_vcpu_second": profile.per_vcpu_second,
              "note": "verify config/cloud_pricing.json against current provider prices"}

    files = sorted({f for pattern in args.telemetry for f in glob.glob(pattern)})
    runs = [r for f in files for r in read_log(f) if r.get("status", "ok") == "ok"]
    print(f"{len(runs)} successful runs from {len(files)} telemetry file(s)")

    # 2. pricing model vs provider, per run
    model, cloud = [], []
    per_batch = defaultdict(lambda: {"runs": 0, "cloud_cost": 0.0, "model_cost": 0.0, "executor_s": 0.0})
    for r in runs:
        c = profile.run_cost(r.get("executor_seconds"))
        model.append(r.get("actual_cost"))
        cloud.append(c)
        b = per_batch[str(r.get(args.batch_field))]
        b["runs"] += 1
        b["cloud_cost"] += c or 0.0
        b["model_cost"] += r.get("actual_cost") or 0.0
        b["executor_s"] += r.get("executor_seconds") or 0.0
    report["model_vs_cloud_per_run"] = spearman(model, cloud)

    # 1. pre-execution prediction vs provider cost, per plan
    if args.measurements:
        rows = []
        for m in load_measurements(Path(args.measurements)):
            for c in m.measured_pool():
                if c.pred_cost is None or not c.executor_seconds:
                    continue
                cloud_cost = profile.run_cost(sorted(c.executor_seconds)[len(c.executor_seconds) // 2])
                rows.append({"query_id": m.query_id, "plan_id": c.plan_id,
                             "predicted_cost": c.pred_cost, "cloud_cost": cloud_cost,
                             "model_actual_cost": c.actual_cost})
        write_csv(out_dir / "predicted_vs_cloud.csv", rows)
        report["predicted_vs_cloud_per_plan"] = spearman(
            [r["predicted_cost"] for r in rows], [r["cloud_cost"] for r in rows])
        if len(rows) >= 3:
            scatter([r["predicted_cost"] for r in rows], [r["cloud_cost"] for r in rows], None,
                    "MOCAP predicted cost (model units)", f"{profile.provider} cost (USD)",
                    "Predicted vs provider cost per plan", out_dir / "predicted_vs_cloud")

    batches = [{"batch": k, **v} for k, v in sorted(per_batch.items())]

    # 3. telemetry vs real bill, per batch
    if args.bill:
        billed = {}
        with open(args.bill, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                billed[str(row[args.bill_batch_col])] = float(row[args.bill_cost_col])
        for b in batches:
            b["billed_cost"] = billed.get(b["batch"])
        matched = [b for b in batches if b.get("billed_cost") is not None]
        report["telemetry_vs_bill_per_batch"] = spearman(
            [b["cloud_cost"] for b in matched], [b["billed_cost"] for b in matched])
        report["unmatched_bill_batches"] = sorted(set(billed) - {b["batch"] for b in batches})
        if matched:
            scatter([b["cloud_cost"] for b in matched], [b["billed_cost"] for b in matched],
                    [b["batch"] for b in matched],
                    "telemetry-derived cost (USD)", "billed cost (USD)",
                    f"MOCAP telemetry vs real {profile.provider} bill", figures / "real_cost")
            print(f"Figure: {figures / 'real_cost.pdf'}")
    else:
        print("No --bill given: skipping the real-invoice comparison (needs a cluster run).")

    write_csv(out_dir / "batches.csv", batches)
    (out_dir / "cloud_cost_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
