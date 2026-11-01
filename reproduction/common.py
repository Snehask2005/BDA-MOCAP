"""
Shared helpers for the 1-click reproduction scripts (owner: Anaswara).

Every paper table / figure for Section 5 is regenerated from the raw
cached measurements in results/phase5/<tag>/measurements and
results/phase6/<tag>/measurements, so a reviewer needs no cluster to
rebuild them -- only to re-measure.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "paper" / "tables"
FIGURES = ROOT / "paper" / "figures"

PRETTY = {
    "mocap": "MOCAP (conformal)",
    "mocap_point": "MOCAP (point filter)",
    "greedy_fastest": "Greedy Fastest",
    "greedy_cheapest": "Greedy Cheapest",
    "random": "Random",
    "oracle": "Oracle",
    "spark_default_cbo_off_aqe_off": "Spark (CBO off, AQE off)",
    "spark_default_cbo_on_aqe_off": "Spark (CBO on, AQE off)",
    "spark_default_cbo_off_aqe_on": "Spark default (CBO off, AQE on)",
    "spark_default_cbo_on_aqe_on": "Spark (CBO on, AQE on)",
}
ORDER = list(PRETTY)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Missing results file {path}. Run the experiment first (see README).")
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value: str, digits: int = 3, scale: float = 1.0) -> str:
    try:
        x = float(value) * scale
    except (TypeError, ValueError):
        return "--"
    if x != x:  # nan
        return "--"
    return f"{x:.{digits}f}"


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(name: str, caption: str, label: str, header: Sequence[str],
                rows: Sequence[Sequence[str]], align: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
    ]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path = TABLES / f"{name}.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def sort_strategies(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda r: ORDER.index(r["strategy"]) if r["strategy"] in ORDER else 99)


def phase5(tag: str) -> Path:
    return ROOT / "results" / "phase5" / tag


def phase6(tag: str) -> Path:
    return ROOT / "results" / "phase6" / tag


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / script), *args]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def tag_arg(argv=None) -> str:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="results sub-folder, e.g. tpcds_sf10")
    return parser.parse_args(argv).tag
