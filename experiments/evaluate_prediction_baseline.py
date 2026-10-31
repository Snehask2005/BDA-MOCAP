import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


INPUT_FILE = Path(
    "results/candidate_repeated_execution/"
    "candidate_execution_summary.csv"
)

OUTPUT_FILE = Path(
    "results/candidate_repeated_execution/"
    "prediction_baseline_metrics.csv"
)


def evaluate(name, predicted, actual):

    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)

    pearson = pearsonr(predicted, actual)
    spearman = spearmanr(predicted, actual)

    mae = np.mean(np.abs(predicted - actual))

    # MAPE is included for completeness, but can become
    # misleading when actual values are very small.
    mape = (
        np.mean(
            np.abs(
                (predicted - actual) / actual
            )
        )
        * 100
    )

    return {
        "target": name,
        "n": len(predicted),
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
        "mae": mae,
        "mape_percent": mape,
    }


def main():

    with INPUT_FILE.open(newline="") as file:
        rows = list(csv.DictReader(file))

    predicted_latency = [
        float(row["predicted_latency"])
        for row in rows
    ]

    actual_latency = [
        float(row["actual_latency_median"])
        for row in rows
    ]

    predicted_cost = [
        float(row["predicted_cost"])
        for row in rows
    ]

    actual_cost = [
        float(row["actual_cost_median"])
        for row in rows
    ]

    results = [
        evaluate(
            "latency",
            predicted_latency,
            actual_latency,
        ),
        evaluate(
            "cost",
            predicted_cost,
            actual_cost,
        ),
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print("=" * 80)
    print("UNCALIBRATED PREDICTION BASELINE")
    print("=" * 80)

    for result in results:

        print(
            f"\nTarget: {result['target']}"
        )

        print(
            f"Samples          : {result['n']}"
        )

        print(
            f"Pearson r        : "
            f"{result['pearson_r']:.6f}"
        )

        print(
            f"Pearson p        : "
            f"{result['pearson_p']:.6f}"
        )

        print(
            f"Spearman rho     : "
            f"{result['spearman_rho']:.6f}"
        )

        print(
            f"Spearman p       : "
            f"{result['spearman_p']:.6f}"
        )

        print(
            f"MAE              : "
            f"{result['mae']:.10f}"
        )

        print(
            f"MAPE             : "
            f"{result['mape_percent']:.2f}%"
        )

    print("\n" + "=" * 80)

    print(
        f"\nMetrics saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
