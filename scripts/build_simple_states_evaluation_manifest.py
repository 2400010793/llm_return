"""Build a simple_states evaluation manifest from one or more regression manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regression_manifests", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--only-existing", action="store_true")
    args = parser.parse_args()

    rows = []
    seen_predictions: set[Path] = set()
    factor_sources: dict[str, Path] = {}
    for regression_manifest in args.regression_manifests:
        with regression_manifest.open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle, delimiter="\t"))
        for source in source_rows:
            report = Path(source["output"])
            predictions = report.with_suffix(".stock_day_predictions.parquet")
            if args.only_existing and not predictions.is_file():
                continue
            resolved_predictions = predictions.resolve()
            if resolved_predictions in seen_predictions:
                continue
            factor_id = report.stem
            prior_source = factor_sources.get(factor_id)
            if prior_source is not None and prior_source != resolved_predictions:
                raise ValueError(
                    f"factor_id collision for {factor_id!r}: "
                    f"{prior_source} and {resolved_predictions}"
                )
            seen_predictions.add(resolved_predictions)
            factor_sources[factor_id] = resolved_predictions
            rows.append({
                "task_id": len(rows),
                "source_task_id": int(source["task_id"]),
                "predictions": str(predictions),
                "factor_id": factor_id,
                "output_dir": str(args.output_root / factor_id),
            })
    if not rows:
        raise ValueError("no regression predictions selected for evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": str(args.regression_manifests[0]),
        "sources": [str(path) for path in args.regression_manifests],
        "output": str(args.output),
        "output_root": str(args.output_root),
        "only_existing": args.only_existing,
        "tasks": len(rows),
        "source_task_ids": [row["source_task_id"] for row in rows],
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
