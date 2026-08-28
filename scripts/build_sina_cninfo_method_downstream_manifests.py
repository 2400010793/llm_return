"""Build first-stage classification/regression manifests for Sina CNINFO-style pools."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("roberta", "bge_m3", "ckip_bert", "xlm_roberta_large")
VARIANTS = ("plain", "short", "masked_short", "long", "masked_long")


def complete(root: Path, model: str, variant: str, shards: int) -> bool:
    return all(
        all((root / f"shard-{shard}" / model / variant / name).is_file() for name in (
            "summary.json", "metadata.jsonl", "short_pooling.npz",
        ))
        for shard in range(shards)
    )


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-root", type=Path,
        default=Path("/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/embeddings"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1"),
    )
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--classification-manifest", type=Path, required=True)
    parser.add_argument("--regression-manifest", type=Path, required=True)
    args = parser.parse_args()
    for model in MODELS:
        for variant in VARIANTS:
            if not complete(args.embedding_root, model, variant, args.expected_shards):
                raise ValueError(f"incomplete embedding output for {model}/{variant}")

    classification: list[dict[str, object]] = []
    regression: list[dict[str, object]] = []
    for model in MODELS:
        for variant in VARIANTS:
            features = ("full_mean",) if variant == "plain" else (
                "title_mean", "body_mean", "full_mean",
            )
            for feature in features:
                for reducer, components in (("none", 0), ("pca", 128)):
                    output = args.output_root / "classification" / (
                        f"{model}_{variant}_{feature}_logistic_{reducer}_{components}.json"
                    )
                    classification.append({
                        "task_id": len(classification), "embedding_root": str(args.embedding_root),
                        "model": model, "variant": variant, "feature": feature,
                        "classifier": "logistic", "reducer": reducer, "components": components,
                        "output": str(output), "train_target": "event_return_3d",
                        "evaluation_target": "next_day_return",
                    })
            for reducer, components in (("none", 0), ("pca", 128)):
                output = args.output_root / "regression" / (
                    f"{model}_{variant}_full_mean_next_day_return_ridge_{reducer}_{components}.json"
                )
                regression.append({
                    "task_id": len(regression), "embedding_root": str(args.embedding_root),
                    "model": model, "variant": variant, "feature": "full_mean",
                    "target": "next_day_return", "regressor": "ridge",
                    "reducer": reducer, "components": components, "output": str(output),
                })
    write_tsv(args.classification_manifest, classification)
    write_tsv(args.regression_manifest, regression)
    summary = {
        "classification_tasks": len(classification),
        "regression_tasks": len(regression),
        "classification_manifest": str(args.classification_manifest),
        "regression_manifest": str(args.regression_manifest),
        "classification_protocol": "6y fit / 2y validation / 1y OOS; 3d label train, next-day evaluation",
        "regression_protocol": "6y fit / 2y validation / 1y OOS; next-day Ridge",
    }
    args.classification_manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
