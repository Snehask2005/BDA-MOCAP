"""
Summarize results/adaptive_policy_experiment.csv.

Reads stored experiment results only; every number printed or saved here
is computed from that file. Costs are local model-derived values
(wall-clock * cores priced via PricingConfig), not cloud billing.

Run:  python summarize_adaptive_policy.py
"""

import csv
import statistics
from collections import defaultdict

INPUT = "results/adaptive_policy_experiment.csv"
OUTPUT = "results/adaptive_policy_summary.csv"

SCENARIO_ORDER = {"relaxed": 0, "moderate": 1, "strict": 2}

FIELDNAMES = [
    "scenario", "policy", "tolerance", "n", "budget",
    "intervention_rate", "within_budget_rate",
    "median_total_cost", "max_total_cost", "median_wall_s",
    "cancelled_attempts", "exactly_one_ok_each",
    "median_cost_vs_strict_policy",
]


def main() -> None:
    with open(INPUT, newline="") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["policy"], r["tolerance"])].append(r)

    # Median cost of the plain strict policy per scenario (comparison base).
    strict_median = {}
    for (scenario, policy, _tol), rs in groups.items():
        if policy == "strict":
            strict_median[scenario] = statistics.median(
                float(r["total_cost"]) for r in rs
            )

    ordered = sorted(
        groups.items(),
        key=lambda kv: (SCENARIO_ORDER.get(kv[0][0], 9), kv[0][1], kv[0][2]),
    )

    summary = []

    for (scenario, policy, tol), rs in ordered:
        n = len(rs)
        totals = [float(r["total_cost"]) for r in rs]
        walls = [float(r["wall_time_s"]) for r in rs]

        intervened = sum(
            1 for r in rs if r["adaptive_action"] not in ("none", "")
        )
        within = sum(int(r["within_budget"]) for r in rs)
        cancelled = sum(int(r["cancelled_rows"]) for r in rs)
        one_ok = all(int(r["ok_rows"]) == 1 for r in rs)

        base = strict_median.get(scenario)
        ratio = (
            statistics.median(totals) / base
            if base
            else ""
        )

        summary.append(
            {
                "scenario": scenario,
                "policy": policy,
                "tolerance": tol,
                "n": n,
                "budget": rs[0]["budget"],
                "intervention_rate": round(intervened / n, 3),
                "within_budget_rate": round(within / n, 3),
                "median_total_cost": statistics.median(totals),
                "max_total_cost": max(totals),
                "median_wall_s": round(statistics.median(walls), 2),
                "cancelled_attempts": cancelled,
                "exactly_one_ok_each": int(one_ok),
                "median_cost_vs_strict_policy": (
                    round(ratio, 2) if ratio != "" else ""
                ),
            }
        )

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary)

    header = (
        f"{'scenario':9s} {'policy':8s} {'tol':>4s} {'n':>2s} "
        f"{'interv':>6s} {'in-budget':>9s} {'median cost':>12s} "
        f"{'vs strict':>9s} {'cancelled':>9s} {'1 ok':>4s}"
    )
    print()
    print(header)
    print("-" * len(header))

    for s in summary:
        print(
            f"{s['scenario']:9s} {s['policy']:8s} "
            f"{str(s['tolerance']):>4s} {s['n']:>2d} "
            f"{s['intervention_rate']:>6.2f} "
            f"{s['within_budget_rate']:>9.2f} "
            f"{s['median_total_cost']:>12.8f} "
            f"{str(s['median_cost_vs_strict_policy']):>9s} "
            f"{s['cancelled_attempts']:>9d} "
            f"{s['exactly_one_ok_each']:>4d}"
        )

    adaptive = [r for r in rows if r["policy"] == "adaptive"]
    intervened = [r for r in adaptive if r["adaptive_action"] not in ("none", "")]
    rescued = [r for r in intervened if int(r["within_budget"]) == 1]

    print()
    print(
        f"Adaptive runs: {len(adaptive)} | intervened: {len(intervened)} | "
        f"intervened AND finished within budget: {len(rescued)}"
    )
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()