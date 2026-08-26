import csv
from pathlib import Path

from scripts.build_next1_comparison_manifest import BASELINES, build_rows


def test_comparison_manifest_adds_pending_strong_predictions(tmp_path: Path) -> None:
    manifest = tmp_path / "strong.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("classifier", "model", "variant", "feature", "output"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow({
            "classifier": "extra_trees", "model": "roberta",
            "variant": "short", "feature": "body_mean",
            "output": "reports/example.json",
        })
    rows = build_rows(manifest)
    assert len(rows) == len(BASELINES) + 1
    assert rows[-1]["model"] == "extra_trees__roberta__short__body_mean"
    assert rows[-1]["path"] == "reports/example.predictions.parquet"
