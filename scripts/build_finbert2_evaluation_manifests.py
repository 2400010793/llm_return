"""Build focused paper-replication and pooling manifests for FinBERT2-base."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


POOLING_FEATURES = (
    "prompt_mean",
    "title_mean",
    "body_mean",
    "title_body_mean",
    "full_mean",
    "cls",
    "full_max",
)


def write_tsv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def classification_rows(
    embedding_root: Path, report_root: Path, variants: tuple[str, ...], features: tuple[str, ...]
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for variant in variants:
        for feature in features:
            rows.append((
                len(rows), "finbert2", embedding_root, "finbert2_base", variant,
                feature, "logistic", "none", 0,
                report_root / f"finbert2_base_{variant}_{feature}_logistic.json",
                "next_day_return", "next_day_return",
            ))
    return rows


def regression_rows(
    embedding_root: Path, report_root: Path, variants: tuple[str, ...], features: tuple[str, ...]
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for variant in variants:
        for feature in features:
            rows.append((
                len(rows), embedding_root, "finbert2_base", variant, feature,
                "next_day_open_to_open_return", "ridge", "final-test", "none", 0,
                report_root / f"finbert2_base_{variant}_{feature}_ridge.json",
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-root", type=Path,
        default=Path("data/processed/pooled_finbert2_embeddings_2010_2026_v1"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("configs/generated"))
    args = parser.parse_args()
    classification_header = (
        "task_id", "phase", "embedding_root", "model", "variant", "feature",
        "classifier", "reducer", "components", "output", "train_target",
        "evaluation_target",
    )
    regression_header = (
        "task_id", "embedding_root", "model", "variant", "feature", "target",
        "regressor", "run_mode", "reducer", "components", "output",
    )
    cls_root = Path("reports/classification/pooled_embeddings/finbert2")
    reg_root = Path("reports/regression/pooled_embeddings/finbert2")
    outputs = {
        "paper_classification": args.output_dir / "finbert2_paper_classification.tsv",
        "pooling_classification": args.output_dir / "finbert2_pooling_classification.tsv",
        "paper_regression": args.output_dir / "finbert2_paper_regression.tsv",
        "pooling_regression": args.output_dir / "finbert2_pooling_regression.tsv",
    }
    write_tsv(
        outputs["paper_classification"], classification_header,
        classification_rows(args.embedding_root, cls_root, ("plain",), ("full_mean",)),
    )
    write_tsv(
        outputs["pooling_classification"], classification_header,
        classification_rows(
            args.embedding_root, cls_root, ("short", "masked_short"), POOLING_FEATURES
        ),
    )
    write_tsv(
        outputs["paper_regression"], regression_header,
        regression_rows(args.embedding_root, reg_root, ("plain",), ("full_mean",)),
    )
    write_tsv(
        outputs["pooling_regression"], regression_header,
        regression_rows(
            args.embedding_root, reg_root, ("short", "masked_short"), POOLING_FEATURES
        ),
    )
    for name, path in outputs.items():
        rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
        print(f"{name}={path} rows={rows}")


if __name__ == "__main__":
    main()
