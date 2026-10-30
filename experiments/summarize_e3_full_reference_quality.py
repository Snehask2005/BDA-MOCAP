import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/e3/quality_validation"

QUERIES = [
    ("q06", "q06_full_reference_summary.csv"),
    ("q07", "q07_full_reference_summary.csv"),
    ("q08", "q08_full_reference_summary.csv"),
    ("q09", "q09_full_reference_summary.csv"),
    ("q10", "q10_full_reference_summary.csv"),
]

OUTPUT = RESULTS / "q06_q10_full_reference_summary.csv"

rows = []
for query_tag, filename in QUERIES:
    path = RESULTS / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing result: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))

    if len(source_rows) != 1:
        raise ValueError(f"Expected exactly one summary row in {path}")

    source = source_rows[0]
    rows.append({
        "query_tag": query_tag,
        **source,
    })

fields = list(rows[0].keys())
with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print("===== Q06–Q10 FULL BEAM-REFERENCE QUALITY =====")
for row in rows:
    print(
        f"{row['query_file']:18} "
        f"reference={row['reference_plan_count']:>3} "
        f"cap48={row['capped_plan_count']:>3} "
        f"valid={row['plans_with_three_valid_measurements']:>3} "
        f"loss={row['observed_pruning_loss_percent']:>6}% "
        f"errors={row['execution_errors']} "
        f"mismatches={row['result_mismatches']}"
    )

print(f"\nSaved combined summary: {OUTPUT}")
print("Scope: uncapped beam-width-40 reference; not exhaustive search.")
