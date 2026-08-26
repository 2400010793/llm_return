"""Build additional scalable classifier tasks for the Sina Prompt-token study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_sina_cninfo_method_prompt_token_classification_manifest import (
    MODELS,
    VARIANTS,
    complete,
    rows_in_complete_shards,
)


CLASSIFIERS = ("linear_svm", "sgd")


def build_rows(
    embedding_root: Path,
    output_root: Path,
    *,
    expected_shards: int = 4,
    expected_rows: int = 4928,
) -> list[dict[str, object]]:
    """Build one PCA128 cell for each model, variant, and classifier."""
    embedding_root = embedding_root.resolve()
    output_root = output_root.resolve()
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for variant in VARIANTS:
            if not complete(embedding_root, model, variant, expected_shards):
                raise ValueError(
                    f"incomplete Prompt-token output for {model}/{variant}"
                )
            observed_rows = rows_in_complete_shards(
                embedding_root, model, variant, expected_shards,
            )
            if observed_rows != expected_rows:
                raise ValueError(
                    f"unexpected row count for {model}/{variant}: "
                    f"{observed_rows} != {expected_rows}"
                )
            for classifier in CLASSIFIERS:
                output = output_root / "prompt_token_classification_other" / (
                    f"{model}_{variant}_prompt_tokens_flat_"
                    f"{classifier}_pca_128.json"
                )
                rows.append({
                    "task_id": len(rows),
                    "embedding_root": str(embedding_root),
                    "model": model,
                    "variant": variant,
                    "feature": "prompt_tokens_flat",
                    "classifier": classifier,
                    "reducer": "pca",
                    "components": 128,
                    "output": str(output),
                    "train_target": "event_return_3d",
                    "evaluation_target": "next_day_return",
                })
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty Prompt-token classifier manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-root", type=Path,
        default=Path(
            "/home/gaozh/news_content_quality_20260812/"
            "cninfo_method_v1/embeddings"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(
            "/home/gaozh/news_content_quality_20260812/"
            "cninfo_method_v1"
        ),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(
            "configs/generated/"
            "sina_cninfo_method_prompt_token_other_classification_v1.tsv"
        ),
    )
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-rows", type=int, default=4928)
    args = parser.parse_args()
    if args.expected_shards < 1 or args.expected_rows < 1:
        raise ValueError("expected-shards and expected-rows must be positive")

    rows = build_rows(
        args.embedding_root,
        args.output_root,
        expected_shards=args.expected_shards,
        expected_rows=args.expected_rows,
    )
    write_manifest(args.manifest, rows)
    summary = {
        "tasks": len(rows),
        "models": list(MODELS),
        "variants": list(VARIANTS),
        "classifiers": list(CLASSIFIERS),
        "feature": "prompt_tokens_flat",
        "reducer": "pca",
        "components": 128,
        "train_target": "event_return_3d",
        "evaluation_target": "next_day_return",
        "protocol": "6y fit / 2y validation / 1y OOS; 2023-2026 test folds",
        "embedding_root": str(args.embedding_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "manifest": str(args.manifest),
    }
    args.manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
