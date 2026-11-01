"""
E5 -- "MOCAP beats Greedy" case study (owner: Anaswara, Phase 5).

Acceptance criterion: at least one query family where MOCAP statistically
outperforms Greedy. For every family and every budget multiplier this
script runs paired Wilcoxon tests of MOCAP vs Greedy Fastest and Greedy
Cheapest on latency, and compares budget-violation counts. It then writes a
short markdown report with the strongest family and concrete per-query
examples where the two picked different plans (predicted vs measured), and
it reports honestly when no family reaches significance.

Reuses the measurements cached by E1 (same --tag / data settings).

Outputs (results/phase5/<tag>/): e5_case_study.csv, e5_examples.csv, e5_case_study.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.cli import add_common_args, get_measurements, write_csv, write_run_info  # noqa: E402
from execution.harness import evaluate  # noqa: E402
from execution.stats import fmt, wilcoxon_paired  # noqa: E402

GREEDY = ("greedy_fastest", "greedy_cheapest")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level")
    args = parser.parse_args(argv)

    env, measurements, out_dir = get_measurements(args, "e5_case_study")
    exp = env.experiment
    delta = args.delta or float(exp.get("default_delta", 0.05))
    multipliers = [float(x) for x in exp["budget_multipliers"]]
    usable = [m for m in measurements if m.measured_pool()]
    if len(usable) < 2:
        print("E5 needs at least 2 measured queries.")
        return 1

    outcomes = [r for r in evaluate(usable, multipliers, [delta], seed=env.seed) if r["plan_id"]]
    index = {(r["query_id"], r["budget_multiplier"], r["strategy"]): r for r in outcomes}
    families = sorted({r["family"] for r in outcomes})

    rows, examples = [], []
    for family in families + ["ALL"]:
        for mult in multipliers:
            qids = sorted({r["query_id"] for r in outcomes
                           if (family == "ALL" or r["family"] == family) and r["budget_multiplier"] == mult})
            for greedy in GREEDY:
                pairs = [(index.get((q, mult, "mocap")), index.get((q, mult, greedy))) for q in qids]
                pairs = [(a, b) for a, b in pairs if a and b]
                if not pairs:
                    continue
                test = wilcoxon_paired([a["latency_s"] for a, _ in pairs], [b["latency_s"] for _, b in pairs])
                mocap_viol = sum(bool(a["budget_violated"]) for a, _ in pairs)
                greedy_viol = sum(bool(b["budget_violated"]) for _, b in pairs)
                faster = test["p_value"] < args.alpha and test["median_diff"] < 0
                safer = mocap_viol < greedy_viol
                rows.append({
                    "family": family, "budget_multiplier": mult, "delta": delta, "greedy": greedy,
                    "n_queries": len(pairs), **{k: test[k] for k in
                                                ("wins", "losses", "ties", "median_diff", "p_value")},
                    "mocap_violations": mocap_viol, "greedy_violations": greedy_viol,
                    "mocap_significantly_faster": faster,
                    "mocap_fewer_violations": safer,
                })
                if family != "ALL":
                    for a, b in pairs:
                        if a["plan_id"] != b["plan_id"]:
                            examples.append({
                                "family": family, "query_id": a["query_id"], "budget_multiplier": mult,
                                "greedy": greedy, "budget": a["budget"],
                                "mocap_plan": a["plan_strategy"], "greedy_plan": b["plan_strategy"],
                                "mocap_pred_cost": a["pred_cost"], "greedy_pred_cost": b["pred_cost"],
                                "mocap_cost": a["cost"], "greedy_cost": b["cost"],
                                "mocap_latency_s": a["latency_s"], "greedy_latency_s": b["latency_s"],
                                "mocap_violated": a["budget_violated"], "greedy_violated": b["budget_violated"],
                            })

    write_csv(out_dir / "e5_case_study.csv", rows)
    write_csv(out_dir / "e5_examples.csv", examples)
    write_run_info(out_dir / "e5_run_info.json", env, args, {"delta": delta})

    wins = [r for r in rows if r["family"] != "ALL"
            and (r["mocap_significantly_faster"] or r["mocap_fewer_violations"])]
    wins.sort(key=lambda r: (not r["mocap_significantly_faster"], r["p_value"],
                             r["mocap_violations"] - r["greedy_violations"]))
    losses = [r for r in rows if r["family"] != "ALL" and r["p_value"] < args.alpha and r["median_diff"] > 0]

    lines = ["# E5 case study: MOCAP vs Greedy", "",
             f"delta = {delta}, significance level = {args.alpha}, queries = {len(usable)}", ""]
    if wins:
        best = wins[0]
        lines += [f"**Strongest family:** `{best['family']}` at budget multiplier {best['budget_multiplier']} "
                  f"vs `{best['greedy']}`: MOCAP faster on {best['wins']} queries, slower on "
                  f"{best['losses']}, tied on {best['ties']} (Wilcoxon p = {fmt(best['p_value'], 4)}); "
                  f"budget violations MOCAP {best['mocap_violations']} vs Greedy {best['greedy_violations']}.", ""]
    else:
        lines += ["**No family met the criterion** (significantly faster or fewer violations). "
                  "Report this honestly; consider more queries per family or larger data.", ""]
    if losses:
        lines += ["**Families where Greedy was significantly faster (report these too):**", ""]
        lines += [f"- `{r['family']}` m={r['budget_multiplier']} vs `{r['greedy']}` (p = {fmt(r['p_value'], 4)})"
                  for r in losses]
        lines.append("")
    lines += ["| family | m | vs | n | W/L/T | p | viol MOCAP/Greedy |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['family']} | {r['budget_multiplier']} | {r['greedy']} | {r['n_queries']} | "
                     f"{r['wins']}/{r['losses']}/{r['ties']} | {fmt(r['p_value'], 4)} | "
                     f"{r['mocap_violations']}/{r['greedy_violations']} |")
    (out_dir / "e5_case_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"Results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
