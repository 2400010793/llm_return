"""Build complete-row manifests for phased pooled stock-day regressions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_pooled_classification_manifest import ROOTS, completed_rows
from src.data.pooled_embeddings import discover_pooled_parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("roberta", "bge_m3", "ckip_bert", "xlm_roberta_large"),
        required=True,
    )
    parser.add_argument(
        "--embedding-root", type=Path, default=None,
        help="Override the configured embedding root for every selected variant.",
    )
    parser.add_argument("--variants", required=True)
    parser.add_argument("--feature", default="title_body_full_concat")
    parser.add_argument(
        "--targets", default="next_day_open_to_open_return",
        help="Continuous one-day target; the paper-faithful default is open-to-open.",
    )
    parser.add_argument(
        "--regressors", default="ridge",
        help="Comma-separated ridge, elasticnet_sgd, huber_sgd, small_mlp.",
    )
    parser.add_argument("--run-mode", choices=("screen", "final-test"), default="screen")
    parser.add_argument(
        "--reducers", default="none,pca:128",
        help="Comma-separated none or pca:N specifications.",
    )
    parser.add_argument("--expected-rows", type=int, default=350577)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/regression/pooled_embeddings/stock_day"),
        help="Directory for regression JSON outputs; keeps independent studies from colliding.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    regressors = [value.strip() for value in args.regressors.split(",") if value.strip()]
    allowed_regressors = {"ridge", "elasticnet_sgd", "huber_sgd", "small_mlp"}
    if not regressors or set(regressors).difference(allowed_regressors):
        raise ValueError(f"unknown regressors: {regressors}")
    reducers: list[tuple[str, int]] = []
    for raw_reducer in (value.strip() for value in args.reducers.split(",")):
        if raw_reducer == "none":
            reducers.append(("none", 0))
        elif raw_reducer.startswith("pca:"):
            components = int(raw_reducer.split(":", 1)[1])
            if components < 1:
                raise ValueError("PCA components must be positive")
            reducers.append(("pca", components))
        elif raw_reducer:
            raise ValueError(f"unknown reducer specification: {raw_reducer}")
    if not reducers:
        raise ValueError("at least one reducer is required")
    if set(variants).difference(ROOTS):
        raise ValueError(f"unknown variants: {variants}")
    if set(targets).difference((
        "next_day_return", "next_day_open_to_open_return",
        "next_day_open_to_open_market_residual",
        "next_day_open_to_open_cs_zscore",
        "next_day_open_to_open_winsor_residual",
        "return_1", "return_3", "return_5", "return_20",
    )):
        raise ValueError(f"unknown targets: {targets}")

    rows: list[dict[str, object]] = []
    for variant in variants:
        embedding_root = args.embedding_root or ROOTS[variant]
        parts = discover_pooled_parts(embedding_root, args.model, variant)
        if not parts or (args.expected_rows and completed_rows(parts) != args.expected_rows):
            raise ValueError(
                f"incomplete embeddings for {args.model}/{variant}: "
                f"{completed_rows(parts) if parts else 0}/{args.expected_rows}"
            )
        for target in targets:
            for regressor in regressors:
                for reducer, components in reducers:
                    stem = "_".join((
                        args.model, variant, args.feature, target, regressor,
                        reducer, str(components), args.run_mode,
                    ))
                    output = args.output_dir / f"{stem}.json"
                    if output.is_file():
                        continue
                    rows.append({
                        "task_id": len(rows),
                        "embedding_root": str(embedding_root),
                        "model": args.model,
                        "variant": variant,
                        "feature": args.feature,
                        "target": target,
                        "regressor": regressor,
                        "run_mode": args.run_mode,
                        "reducer": reducer,
                        "components": components,
                        "output": str(output),
                    })
    if not rows:
        raise ValueError("no pending regression tasks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": args.model, "variants": variants, "feature": args.feature,
        "targets": targets, "regressors": regressors, "run_mode": args.run_mode,
        "reducers": reducers, "tasks": len(rows), "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
