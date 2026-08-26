"""Build the bounded O2O classification manifest.

The manifest reuses the complete pooled embeddings and changes only the
supervised target to the strictly aligned next-open-to-next-open return.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import discover_pooled_parts


EMBEDDING_ROOT = Path("data/processed/pooled_embeddings_v3")
MODELS = ("roberta", "bge_m3")
VARIANTS = ("short", "masked_short")
FEATURES = ("body_mean", "full_mean", "title_body_full_concat")
CLASSIFIERS = ("logistic", "mlp", "hist_gradient_boosting", "extra_trees")
REDUCERS = (("none", 0), ("pca", 128))
TARGET = "next_day_open_to_open_return"


def completed_rows(parts: list[Path]) -> int:
    return sum(
        int(json.loads((part / "summary.json").read_text(encoding="utf-8"))["rows"])
        for part in parts
    )


def build_rows(
    root: Path,
    *,
    include_existing: bool = False,
    expected_rows: int = 350577,
    embedding_root: Path = EMBEDDING_ROOT,
    models: tuple[str, ...] = MODELS,
    variants: tuple[str, ...] = VARIANTS,
    features: tuple[str, ...] = FEATURES,
    classifiers: tuple[str, ...] = CLASSIFIERS,
    reducers: tuple[tuple[str, int], ...] = REDUCERS,
    target: str = TARGET,
    phase: str = "o2o_core",
    output_root: Path = Path("reports/classification/pooled_embeddings/o2o_core"),
    search_stage: str = "coarse",
    seed: int = 42,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    absolute_embedding_root = root / embedding_root
    for model in models:
        for variant in variants:
            parts = discover_pooled_parts(absolute_embedding_root, model, variant)
            rows_found = completed_rows(parts)
            if rows_found != expected_rows:
                raise ValueError(
                    f"incomplete embeddings for {model}/{variant}: "
                    f"{rows_found} != {expected_rows}"
                )
            for feature in features:
                for classifier in classifiers:
                    for reducer, components in reducers:
                        stem = "_".join(
                            (
                                model,
                                variant,
                                feature,
                                classifier,
                                reducer,
                                str(components),
                                f"seed{seed}",
                            )
                        )
                        output = output_root / f"{stem}.json"
                        if not include_existing and (root / output).is_file():
                            continue
                        rows.append(
                            {
                                "task_id": len(rows),
                                "phase": phase,
                                "embedding_root": str(embedding_root),
                                "model": model,
                                "variant": variant,
                                "feature": feature,
                                "classifier": classifier,
                                "reducer": reducer,
                                "components": components,
                                "search_stage": search_stage,
                                "seed": seed,
                                "output": str(output),
                                "train_target": target,
                                "evaluation_target": target,
                            }
                        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/generated/pooled_o2o_classification_seed42.tsv"),
    )
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--expected-rows", type=int, default=350577)
    parser.add_argument("--embedding-root", type=Path, default=EMBEDDING_ROOT)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--features", default=",".join(FEATURES))
    parser.add_argument("--classifiers", default=",".join(CLASSIFIERS))
    parser.add_argument(
        "--reducers",
        default=",".join(
            reducer if not components else f"{reducer}:{components}"
            for reducer, components in REDUCERS
        ),
        help="Comma-separated reducer specifications such as none,pca:128",
    )
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--phase", default="o2o_core")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/classification/pooled_embeddings/o2o_core"),
    )
    parser.add_argument("--search-stage", default="coarse")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    features = tuple(value.strip() for value in args.features.split(",") if value.strip())
    classifiers = tuple(
        value.strip() for value in args.classifiers.split(",") if value.strip()
    )
    reducers: list[tuple[str, int]] = []
    for specification in args.reducers.split(","):
        name, separator, components = specification.strip().partition(":")
        if not name:
            continue
        reducers.append((name, int(components) if separator else 0))

    rows = build_rows(
        args.root,
        include_existing=args.include_existing,
        expected_rows=args.expected_rows,
        embedding_root=args.embedding_root,
        models=models,
        variants=variants,
        features=features,
        classifiers=classifiers,
        reducers=tuple(reducers),
        target=args.target,
        phase=args.phase,
        output_root=args.output_root,
        search_stage=args.search_stage,
        seed=args.seed,
    )
    if not rows:
        raise ValueError("no missing O2O classification cells were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "phase": args.phase,
        "tasks": len(rows),
        "output": str(args.output),
        "embedding_root": str(args.embedding_root),
        "output_root": str(args.output_root),
        "target": args.target,
        "models": list(models),
        "variants": list(variants),
        "features": list(features),
        "classifiers": list(classifiers),
        "reducers": [
            {"name": reducer, "components": components}
            for reducer, components in reducers
        ],
        "seed": args.seed,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
