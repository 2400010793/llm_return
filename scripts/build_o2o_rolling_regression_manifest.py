"""Build a unique manifest for rolling 2018-2026 O2O regressions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_pooled_classification_manifest import ROOTS, completed_rows
from src.data.pooled_embeddings import discover_pooled_parts


def parse_reducers(value: str) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for raw in (item.strip() for item in value.split(",")):
        if not raw:
            continue
        if raw == "none":
            result.append(("none", 0))
        elif raw.startswith("pca:"):
            components = int(raw.split(":", 1)[1])
            if components < 1:
                raise ValueError("PCA components must be positive")
            result.append(("pca", components))
        else:
            raise ValueError(f"unknown reducer: {raw}")
    if not result:
        raise ValueError("at least one reducer is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="roberta,bge_m3")
    parser.add_argument("--variants", default="short,masked_short")
    parser.add_argument("--features", default="body_mean,full_mean,title_body_full_concat")
    parser.add_argument("--regressors", default="ridge,huber_sgd")
    parser.add_argument("--reducers", default="none,pca:128")
    parser.add_argument("--target", default="next_day_open_to_open_return")
    parser.add_argument("--expected-rows", type=int, default=350577)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=None,
        help="Override the pooled embedding root for all variants.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/regression/pooled_embeddings/o2o_rolling_2018_2026"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/generated/o2o_rolling_regression_2018_2026.tsv"),
    )
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    features = [item.strip() for item in args.features.split(",") if item.strip()]
    regressors = [item.strip() for item in args.regressors.split(",") if item.strip()]
    allowed_regressors = {"ridge", "huber_sgd", "small_mlp"}
    if set(models).difference(("roberta", "bge_m3")):
        raise ValueError(f"unknown models: {models}")
    if set(variants).difference(("short", "masked_short")):
        raise ValueError(f"unknown variants: {variants}")
    if set(regressors).difference(allowed_regressors):
        raise ValueError(f"unknown regressors: {regressors}")
    reducers = parse_reducers(args.reducers)

    rows: list[dict[str, object]] = []
    for model in models:
        for variant in variants:
            root = args.embedding_root or ROOTS[variant]
            parts = discover_pooled_parts(root, model, variant)
            count = completed_rows(parts) if parts else 0
            if count != args.expected_rows:
                raise ValueError(f"incomplete embeddings for {model}/{variant}: {count}/{args.expected_rows}")
            for feature in features:
                for regressor in regressors:
                    for reducer, components in reducers:
                        stem = f"{model}_{variant}_{feature}_{regressor}_{reducer}_{components}"
                        output = args.output_root / f"{stem}.json"
                        completed = output.parent / f"{output.stem}.artifacts" / "COMPLETED"
                        if output.is_file() and completed.is_file():
                            continue
                        rows.append({
                            "task_id": len(rows),
                            "embedding_root": str(root),
                            "model": model,
                            "variant": variant,
                            "feature": feature,
                            "target": args.target,
                            "regressor": regressor,
                            "run_mode": "final-test",
                            "reducer": reducer,
                            "components": components,
                            "output": str(output),
                        })
    if not rows:
        raise ValueError("no pending rolling regression tasks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "models": models, "variants": variants, "features": features,
        "regressors": regressors, "reducers": reducers,
        "target": args.target, "expected_rows": args.expected_rows,
        "run_mode": "final-test", "tasks": len(rows),
        "output": str(args.output), "output_root": str(args.output_root),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
