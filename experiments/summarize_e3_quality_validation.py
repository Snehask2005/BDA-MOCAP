import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/e3/quality_validation"

INPUTS = [
    ("q06_pilot", "q06_pilot_summary.csv"),
    ("q07_validation", "q07_validation_summary.csv"),
    ("q08_validation", "q08_validation_summary.csv"),
    ("q09_validation", "q09_validation_summary.csv"),
    ("q10_validation", "q10_validation_summary.csv"),
]

OUTPUT = RESULTS / "q06_q10_quality_summary.csv"

rows = []

for tag, filename in INPUTS:
    path = RESULTS / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation summary: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))

    if len(data) != 1:
        raise ValueError(f"Expected one summary row in {path}")

    rows.append({"validation_tag": tag, **data[0]})

fields = list(rows[0].keys())
with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print("===== Q06–Q10 QUALITY VALIDATION =====")

for row in rows:
    print(
        f"{row['query_file']:18} "
        f"reference={row['reference_candidate_count']:>3} "
        f"cap48={row['capped_candidate_count']:>3} "
        f"retained={row['screening_best_retained']:5} "
        f"finalist_loss={row['finalist_pruning_loss_percent']:>6}% "
        f"screen_errors={row['screen_errors']} "
        f"mismatches={row['screen_result_mismatches']} "
        f"finalist_errors={row['finalist_errors']} "
        f"finalist_mismatches={row['finalist_result_mismatches']}"
    )

print(f"\nSaved combined summary: {OUTPUT}")
print(
    "\nReminder: these results compare repeated finalists against an "
    "uncapped beam-40 reference, not a true exhaustive search."
)
