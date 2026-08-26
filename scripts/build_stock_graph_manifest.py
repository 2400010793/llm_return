"""Build the fixed 3-mode by 9-year stock graph experiment manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODES = ("self", "industry", "random")
YEARS = tuple(range(2018, 2027))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/classification/stock_graph_roberta_masked_v1"),
    )
    args = parser.parse_args()
    rows = []
    for mode in MODES:
        for year in YEARS:
            rows.append({
                "task_id": len(rows),
                "mode": mode,
                "test_year": year,
                "output": str(args.report_root / f"{mode}_test{year}_seed42.json"),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "mode", "test_year", "output"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} graph tasks to {args.output}")


if __name__ == "__main__":
    main()
