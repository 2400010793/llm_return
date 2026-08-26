"""Build staged Slurm manifests for pooled-embedding classification."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import discover_pooled_parts


ROOTS = {
    "plain": Path("data/processed/pooled_paper_hk_embeddings_v1"),
    "short": Path("data/processed/pooled_embeddings_v3"),
    "masked_short": Path("data/processed/pooled_embeddings_v3"),
    "long": Path("data/processed/pooled_long_embeddings_v4"),
    "masked_long": Path("data/processed/pooled_long_embeddings_v4"),
}
CORE_FEATURES = ("title_mean", "body_mean", "full_mean")
# The seven requested representations are all available from the frozen mean
# fields. The final two are loader-side concatenations, so adding them does
# not re-encode either the 2018--2026 rows or the historical increment.
ALL_POOLING_FEATURES = (
    "prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean",
    "title_body_concat", "title_body_full_concat",
)
ALL_CLASSIFIERS = (
    "logistic", "linear_svm", "sgd", "mlp", "simple_mlp", "lstm",
    "random_forest", "extra_trees", "hist_gradient_boosting",
    "xgboost", "lightgbm", "catboost", "knn",
)
PRIMARY_TARGET = "next_day_return"


def phase_cells(phase: str, *, include_unreduced_linear_svm: bool = False):
    if phase == "screening":
        for feature in CORE_FEATURES:
            yield feature, "logistic", "none", 0
            yield feature, "logistic", "pca", 128
        # The prompt representation is all contextualized token vectors, not
        # their mean. Its flattened dimensionality makes PCA mandatory here.
        yield "prompt_tokens_flat", "logistic", "pca", 128
    elif phase == "linear":
        for feature in CORE_FEATURES:
            # Logistic is already completed in screening; this phase adds the
            # two complementary scalable linear decision rules.
            if include_unreduced_linear_svm:
                yield feature, "linear_svm", "none", 0
            yield feature, "linear_svm", "pca", 128
            for reducer, components in (("none", 0), ("pca", 128)):
                yield feature, "sgd", reducer, components
        for classifier in ("linear_svm", "sgd"):
            yield "prompt_tokens_flat", classifier, "pca", 128
    elif phase == "nonlinear":
        for feature in ("body_mean", "full_mean"):
            yield feature, "mlp", "none", 0
            yield feature, "mlp", "pca", 128
            yield feature, "random_forest", "pca", 128
    elif phase == "pca_sweep":
        for feature in ("body_mean", "full_mean"):
            for components in (32, 64, 128, 256):
                yield feature, "logistic", "pca", components
    elif phase == "aggregation":
        # Compare token-weighted pooling with segment-preserving concatenation.
        for feature in ("title_body_mean", "title_body_concat"):
            yield feature, "logistic", "none", 0
            yield feature, "logistic", "pca", 128
        yield "prompt_tokens_title_body_concat", "logistic", "pca", 128
    elif phase == "triple_concat":
        yield "title_body_full_concat", "logistic", "none", 0
        yield "title_body_full_concat", "logistic", "pca", 128
    elif phase == "triple_models":
        if include_unreduced_linear_svm:
            yield "title_body_full_concat", "linear_svm", "none", 0
        yield "title_body_full_concat", "linear_svm", "pca", 128
        for classifier in ("sgd", "mlp"):
            yield "title_body_full_concat", classifier, "none", 0
            yield "title_body_full_concat", classifier, "pca", 128
    elif phase == "complex_pooling":
        for feature in ("title_body_mean", "title_body_concat", "title_body_full_concat"):
            yield feature, "logistic", "none", 0
    elif phase == "complex_pooling_pca":
        for feature in ("title_body_mean", "title_body_concat", "title_body_full_concat"):
            yield feature, "logistic", "pca", 128
    elif phase == "all_pooling":
        for feature in ALL_POOLING_FEATURES:
            # PCA is a separate representation/modeling ablation. This
            # matrix deliberately has one reducer per representation so each
            # requested classifier/representation cell is run exactly once.
            for classifier in ALL_CLASSIFIERS:
                yield feature, classifier, "none", 0
    elif phase == "next1_core":
        for feature in ("body_mean", "full_mean", "title_body_full_concat"):
            for classifier in ("logistic", "mlp"):
                yield feature, classifier, "none", 0
    elif phase == "lstm":
        # The pooled runner's LSTM is a paper-style one-step baseline over
        # frozen row-level embeddings.  Keep it separate from the existing
        # matrix so adding this neural baseline cannot resubmit old cells.
        for feature in ("body_mean", "full_mean", "title_body_full_concat"):
            yield feature, "lstm", "none", 0
    elif phase == "paper_hk":
        # CKX use a frozen article embedding followed by a linear prediction
        # layer.  PCA and a three-layer-style MLP are retained as predeclared
        # extensions rather than silently replacing the paper baseline.
        for classifier in ("logistic", "mlp"):
            yield "full_mean", classifier, "none", 0
            yield "full_mean", classifier, "pca", 128
    elif phase == "temporal_mlp":
        # Re-run both neural baselines under chronological validation-based
        # early stopping.  Keep this matrix separate from historical MLP
        # outputs, whose sklearn internal validation was random within fit.
        for feature in ALL_POOLING_FEATURES:
            for classifier in ("mlp", "simple_mlp"):
                yield feature, classifier, "none", 0
    else:
        raise ValueError(f"unknown phase: {phase}")


def completed_rows(parts: list[Path]) -> int:
    return sum(
        int(json.loads((part / "summary.json").read_text(encoding="utf-8"))["rows"])
        for part in parts
    )


def build_rows(
    root: Path,
    phase: str,
    *,
    include_existing: bool = False,
    expected_rows: int = 350577,
    models: tuple[str, ...] = ("roberta", "bge_m3"),
    variants: tuple[str, ...] = tuple(ROOTS),
    include_prompt_tokens: bool = False,
    include_unreduced_linear_svm: bool = False,
    embedding_root_override: Path | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    for variant, relative_embedding_root in ROOTS.items():
        if variant not in variants:
            continue
        embedding_root = (
            embedding_root_override
            if embedding_root_override is not None
            else root / relative_embedding_root
        )
        for model in models:
            try:
                parts = discover_pooled_parts(embedding_root, model, variant)
            except ValueError as exc:
                print(
                    f"Skipping incomplete {model}/{variant}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not parts or (expected_rows and completed_rows(parts) != expected_rows):
                continue
            for feature, classifier, reducer, components in phase_cells(
                phase, include_unreduced_linear_svm=include_unreduced_linear_svm
            ):
                if not include_prompt_tokens and feature.startswith("prompt_tokens"):
                    continue
                key = (model, variant, feature, classifier, reducer, str(components))
                if key in seen:
                    continue
                seen.add(key)
                stem = "_".join(key)
                output = Path(f"reports/classification/pooled_embeddings/{phase}/{stem}.json")
                if not include_existing and (root / output).is_file():
                    continue
                rows.append({
                    "task_id": len(rows),
                    "phase": phase,
                    "embedding_root": str(
                        embedding_root_override
                        if embedding_root_override is not None
                        else relative_embedding_root
                    ),
                    "model": model,
                    "variant": variant,
                    "feature": feature,
                    "classifier": classifier,
                    "reducer": reducer,
                    "components": components,
                    "output": str(output),
                    # All primary pooled-classification phases use the same
                    # one-day target. The three-day event target is retained
                    # only for explicitly named legacy/ablation manifests.
                    "train_target": PRIMARY_TARGET,
                    "evaluation_target": PRIMARY_TARGET,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--phase",
        choices=(
            "screening", "aggregation", "triple_concat", "triple_models",
            "linear", "nonlinear", "pca_sweep", "complex_pooling", "all_pooling",
            "complex_pooling_pca",
            "next1_core",
            "lstm",
            "paper_hk",
            "temporal_mlp",
        ),
        default="screening",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-existing", action="store_true", help="Include cells whose result JSON already exists")
    parser.add_argument("--expected-rows", type=int, default=350577, help="Require this many completed embedding rows; 0 permits partial probe manifests")
    parser.add_argument("--models", default="roberta,bge_m3", help="Comma-separated model filter")
    parser.add_argument("--variants", default=",".join(ROOTS), help="Comma-separated variant filter")
    parser.add_argument(
        "--embedding-root", type=Path, default=None,
        help="Use one custom pooled-embedding root for every selected variant.",
    )
    parser.add_argument(
        "--include-prompt-tokens",
        action="store_true",
        help="Include very high-dimensional prompt-token cells; disabled by default because long variants exceed normal memory budgets",
    )
    parser.add_argument(
        "--include-unreduced-linear-svm",
        action="store_true",
        help="Diagnostic opt-in for calibrated Linear SVM without PCA; disabled by default because historical cells took up to 40.56 hours",
    )
    args = parser.parse_args()
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    unknown_models = set(models).difference((
        "roberta", "bge_m3", "ckip_bert", "xlm_roberta_large",
    ))
    unknown_variants = set(variants).difference(ROOTS)
    if unknown_models or unknown_variants:
        raise ValueError(f"unknown models={sorted(unknown_models)} variants={sorted(unknown_variants)}")
    rows = build_rows(
        args.root, args.phase,
        include_existing=args.include_existing,
        expected_rows=args.expected_rows,
        models=models,
        variants=variants,
        include_prompt_tokens=args.include_prompt_tokens,
        include_unreduced_linear_svm=args.include_unreduced_linear_svm,
        embedding_root_override=(
            args.embedding_root.resolve() if args.embedding_root is not None else None
        ),
    )
    if not rows:
        raise ValueError("no fully complete pooled embedding combinations were discovered")
    output = args.output or Path(f"configs/pooled_classification_{args.phase}.tsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "phase": args.phase,
        "tasks": len(rows),
        "output": str(output),
        "models": sorted({str(row["model"]) for row in rows}),
        "variants": sorted({str(row["variant"]) for row in rows}),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
