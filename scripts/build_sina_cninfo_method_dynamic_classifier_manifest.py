"""Build dynamic prompt-token gate + classifier tasks from existing embeddings.

Prompt-token tensors are not available for the natural ``plain`` variant in
this embedding study, so the manifest deliberately covers only the four
variants with persisted token-level assets.  No embedding generation occurs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from scripts.build_sina_cninfo_method_prompt_token_classification_manifest import (
        MODELS,
        VARIANTS,
        complete,
        rows_in_complete_shards,
    )
    from scripts.run_pooled_embedding_classification import SUPPORTED_CLASSIFIERS
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from build_sina_cninfo_method_prompt_token_classification_manifest import (
        MODELS,
        VARIANTS,
        complete,
        rows_in_complete_shards,
    )
    from run_pooled_embedding_classification import SUPPORTED_CLASSIFIERS


FEATURE = "prompt_tokens_flat"
REDUCER = "dynamic_token_gate"
CLASSIFIERS = tuple(SUPPORTED_CLASSIFIERS)


def build_rows(
    embedding_root: Path,
    output_root: Path,
    *,
    expected_shards: int = 4,
    expected_rows: int = 4928,
    classifiers: tuple[str, ...] = CLASSIFIERS,
) -> list[dict[str, object]]:
    """Build one rolling dynamic-gate task per model/variant/classifier."""
    if not classifiers:
        raise ValueError("at least one classifier is required")
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
            for classifier in classifiers:
                output = output_root / "prompt_token_dynamic_classification" / (
                    f"{model}_{variant}_{FEATURE}_{classifier}_{REDUCER}.json"
                )
                rows.append({
                    "task_id": len(rows),
                    "embedding_root": str(embedding_root),
                    "model": model,
                    "variant": variant,
                    "feature": FEATURE,
                    "classifier": classifier,
                    "reducer": REDUCER,
                    "components": 0,
                    "output": str(output),
                    "train_target": "event_return_3d",
                    "evaluation_target": "next_day_return",
                })
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty dynamic classifier manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-root", type=Path,
        default=Path(
            "/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/"
            "cninfo_method_v1/embeddings"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(
            "/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/"
            "cninfo_method_v1"
        ),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(
            "configs/generated/"
            "sina_cninfo_method_dynamic_classifier_v1.tsv"
        ),
    )
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-rows", type=int, default=4928)
    parser.add_argument(
        "--classifiers", default=",".join(CLASSIFIERS),
        help="Comma-separated subset of the registered pooled classifiers.",
    )
    args = parser.parse_args()
    if args.expected_shards < 1 or args.expected_rows < 1:
        raise ValueError("expected-shards and expected-rows must be positive")
    classifiers = tuple(value.strip() for value in args.classifiers.split(",") if value.strip())
    unknown = sorted(set(classifiers).difference(CLASSIFIERS))
    if unknown:
        raise ValueError(f"unsupported classifiers: {unknown}")

    rows = build_rows(
        args.embedding_root,
        args.output_root,
        expected_shards=args.expected_shards,
        expected_rows=args.expected_rows,
        classifiers=classifiers,
    )
    write_manifest(args.manifest, rows)
    summary = {
        "tasks": len(rows),
        "models": list(MODELS),
        "variants": list(VARIANTS),
        "feature": FEATURE,
        "reducer": REDUCER,
        "classifiers": list(classifiers),
        "train_target": "event_return_3d",
        "evaluation_target": "next_day_return",
        "protocol": "6y fit / 2y validation / 1y OOS; 2023-2026 test folds",
        "generates_embeddings": False,
        "embedding_root": str(args.embedding_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "manifest": str(args.manifest.resolve()),
    }
    args.manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
