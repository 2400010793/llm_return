"""Paper-style rolling out-of-sample one-day direction classification.

Replicates the paper's timing design as closely as the Chinese panel permits:
8 calendar years in-sample, the first 6 for fitting and the last 2 for
chronological validation/tuning, followed by a 1-year out-of-sample test.
The primary project protocol uses the same one-day return target for fitting,
validation, and out-of-sample evaluation. ``event_return_3d`` is retained only
as an explicit legacy/ablation override. ``--target-column`` remains as a
backward-compatible option that sets both columns to the same target.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_classification import _features, _grid, _truncate_text
from src.evaluation.classification import evaluate_binary_classification
from src.models.representation_models import _classifier
from src.models.dimension_reduction import fit_reduce


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def finite_rows(y: pd.Series) -> np.ndarray:
    return np.isfinite(pd.to_numeric(y, errors="coerce").to_numpy(dtype=float))


def align_precomputed_matrix(
    panel: pd.DataFrame,
    matrix: np.ndarray,
    metadata: pd.DataFrame | None,
    *,
    alignment_key: str = "document_id",
    metadata_row_index_offset: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fail closed when aligning a frozen matrix to a sorted/subset panel.

    Metadata-key alignment permits the matrix to contain rows outside the
    requested panel. Positional alignment is accepted only for equal complete
    row counts and uses ``__panel_position`` captured before panel sorting.
    """
    if metadata is None:
        if len(panel) != len(matrix):
            raise ValueError(
                f"panel and matrix have different row counts: {len(panel)} != {len(matrix)}"
            )
        if "__panel_position" not in panel:
            raise ValueError("positional alignment requires __panel_position")
        positions = pd.to_numeric(panel["__panel_position"], errors="raise").to_numpy(dtype=np.int64)
        if len(np.unique(positions)) != len(positions) or positions.min(initial=0) < 0 or positions.max(initial=-1) >= len(matrix):
            raise ValueError("invalid or duplicate __panel_position values")
        return np.asarray(matrix[positions]), {
            "mode": "exact_positional", "rows": len(panel), "key": "__panel_position",
        }

    if len(metadata) != len(matrix):
        raise ValueError(
            f"metadata and matrix row counts differ: {len(metadata)} != {len(matrix)}"
        )
    if alignment_key not in panel or alignment_key not in metadata:
        raise ValueError(f"alignment key {alignment_key!r} missing from panel or metadata")
    panel_keys = panel[alignment_key].copy()
    metadata_keys = metadata[alignment_key].copy()
    if metadata_row_index_offset:
        if alignment_key != "row_index":
            raise ValueError("metadata_row_index_offset is only valid for row_index")
        metadata_keys = pd.to_numeric(metadata_keys, errors="raise") + metadata_row_index_offset
    if panel_keys.duplicated().any():
        raise ValueError(f"panel alignment key {alignment_key!r} contains duplicates")
    if metadata_keys.duplicated().any():
        raise ValueError(f"metadata alignment key {alignment_key!r} contains duplicates")
    lookup = pd.Series(np.arange(len(metadata_keys), dtype=np.int64), index=metadata_keys)
    positions = lookup.reindex(panel_keys)
    if positions.isna().any():
        missing = panel_keys.iloc[np.flatnonzero(positions.isna().to_numpy())[:5]].tolist()
        raise ValueError(f"metadata does not cover all panel keys; examples={missing}")
    return np.asarray(matrix[positions.to_numpy(dtype=np.int64)]), {
        "mode": "metadata_key",
        "key": alignment_key,
        "rows": len(panel),
        "metadata_rows": len(metadata),
        "metadata_row_index_offset": metadata_row_index_offset,
    }


def evaluate(y: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(probabilities)
    return evaluate_binary_classification(values[mask], probabilities[mask])


def fit_one(
    x_fit,
    y_fit_train,
    x_val,
    y_val_eval,
    x_all,
    y_all_train,
    x_test,
    classifier,
    args,
):
    fit_mask = finite_rows(y_fit_train)
    all_mask = finite_rows(y_all_train)
    y_fit_values = pd.to_numeric(y_fit_train, errors="coerce").to_numpy(dtype=float)[fit_mask]
    labels = (y_fit_values > 0).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("fit window has fewer than two classes")
    params = {}
    grid = _grid(classifier, True, args.search_stage)
    best_score = -np.inf
    # Tune only on the paper's two-year validation window.
    from sklearn.model_selection import ParameterGrid
    for candidate in ParameterGrid(grid):
        model = _classifier(classifier, args.seed, **candidate)
        model.fit(x_fit[fit_mask], labels)
        val_prob = model.predict_proba(x_val)[:, 1]
        metrics = evaluate(y_val_eval, val_prob)
        # The paper tunes the sentiment classifier against out-of-sample
        # classification accuracy; AUC remains a reported supplementary metric.
        score = metrics.get("accuracy", np.nan)
        if not np.isfinite(score):
            score = metrics.get("accuracy", -np.inf)
        if score > best_score:
            best_score, params = score, dict(candidate)
    final = _classifier(classifier, args.seed, **params)
    final.fit(x_all[all_mask], (pd.to_numeric(y_all_train, errors="coerce").to_numpy(dtype=float)[all_mask] > 0).astype(int))
    return final.predict_proba(x_test)[:, 1], params


def reduce_windows(x_fit, x_val, x_all, x_test, args):
    """Fit reducers separately for the fit and all-training windows."""
    if args.reducer == "none":
        return x_fit, x_val, x_all, x_test, None, None
    fit_reduced = fit_reduce(
        x_fit, x_val, method=args.reducer,
        n_components=args.reducer_components, random_state=args.seed,
    )
    all_reduced = fit_reduce(
        x_all, x_test, method=args.reducer,
        n_components=args.reducer_components, random_state=args.seed,
    )
    return (
        fit_reduced.train,
        fit_reduced.predict,
        all_reduced.train,
        all_reduced.predict,
        fit_reduced.reducer,
        all_reduced.reducer,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel")
    parser.add_argument(
        "--train-target-column",
        default="next_day_return",
        help="One-day return target used to fit direction; event_return_3d is legacy-only.",
    )
    parser.add_argument(
        "--evaluation-target-column",
        default="next_day_return",
        help="One-day return target used for validation and out-of-sample accuracy.",
    )
    parser.add_argument(
        "--target-column",
        default="",
        help="Backward-compatible override setting both train and evaluation targets.",
    )
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--representations", default="word_tfidf,char_tfidf,lexicon,word2vec")
    parser.add_argument("--classifiers", default="logistic,random_forest,mlp")
    parser.add_argument("--qwen-npy", default="")
    parser.add_argument("--roberta-npy", default="")
    parser.add_argument("--bge-m3-npy", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-chars", type=int, default=12000)
    parser.add_argument("--search-stage", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--word2vec-size", type=int, default=100)
    parser.add_argument("--word2vec-epochs", type=int, default=10)
    parser.add_argument(
        "--reducer", choices=["none", "svd", "pca"], default="none",
        help="Optional training-window-only dimensionality reduction; SVD for sparse text, PCA for dense embeddings.",
    )
    parser.add_argument("--reducer-components", type=int, default=128)
    parser.add_argument("--output", default="reports/classification/paper_rolling/results.json")
    args = parser.parse_args()
    if args.target_column:
        args.train_target_column = args.target_column
        args.evaluation_target_column = args.target_column
    seed_everything(args.seed)

    frame = pd.read_parquet(args.panel).copy()
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
    frame = frame.dropna(subset=[args.date_column]).sort_values(args.date_column).reset_index(drop=True)
    frame[args.text_column] = frame[args.text_column].map(lambda x: _truncate_text(x, args.max_input_chars))
    frame["year"] = frame[args.date_column].dt.year
    years = sorted(frame["year"].unique())
    if len(years) < 9:
        raise ValueError(f"paper rolling design needs at least 9 calendar years; found {years}")

    representations = [x.strip() for x in args.representations.split(",") if x.strip()]
    if args.qwen_npy:
        representations.append("npy:" + args.qwen_npy)
    if args.roberta_npy:
        representations.append("npy:" + args.roberta_npy)
    if args.bge_m3_npy:
        representations.append("npy:" + args.bge_m3_npy)
    classifiers = [x.strip() for x in args.classifiers.split(",") if x.strip()]
    results: list[dict[str, Any]] = []

    # One-year test folds after an 8-year expanding calendar block.
    for test_year in years[8:]:
        in_years = years[years.index(test_year) - 8:years.index(test_year)]
        fit_years, val_years = in_years[:6], in_years[6:]
        fit = frame[frame.year.isin(fit_years)]
        val = frame[frame.year.isin(val_years)]
        all_train = frame[frame.year.isin(in_years)]
        test = frame[frame.year == test_year]
        if fit.empty or val.empty or test.empty:
            continue
        for representation in representations:
            # Fit text representations only on the six-year fitting sample.
            # Precomputed vectors are positionally aligned with the sorted panel.
            if representation.startswith("npy:"):
                matrix = np.load(representation[4:])
                x_fit = matrix[fit.index.to_numpy()]
                x_val = matrix[val.index.to_numpy()]
                x_all = matrix[all_train.index.to_numpy()]
                x_test = matrix[test.index.to_numpy()]
            else:
                x_fit, x_val = _features(
                    representation,
                    fit[args.text_column].tolist(),
                    val[args.text_column].tolist(),
                    args,
                )
                x_all, x_test = _features(
                    representation,
                    all_train[args.text_column].tolist(),
                    test[args.text_column].tolist(),
                    args,
                )
            x_fit, x_val, x_all, x_test, fit_reducer, all_reducer = reduce_windows(
                x_fit, x_val, x_all, x_test, args
            )
            for classifier in classifiers:
                probabilities, params = fit_one(
                    x_fit,
                    fit[args.train_target_column],
                    x_val,
                    val[args.evaluation_target_column],
                    x_all,
                    all_train[args.train_target_column],
                    x_test,
                    classifier,
                    args,
                )
                metrics = evaluate(test[args.evaluation_target_column], probabilities)
                results.append(
                    {
                        "test_year": int(test_year),
                        "fit_years": fit_years,
                        "validation_years": val_years,
                        "representation": representation,
                        "classifier": classifier,
                        "train_target": args.train_target_column,
                        "evaluation_target": args.evaluation_target_column,
                        "reducer": args.reducer,
                        "reducer_components": args.reducer_components if args.reducer != "none" else None,
                        "fit_explained_variance": (
                            float(np.sum(fit_reducer.explained_variance_ratio_))
                            if fit_reducer is not None and hasattr(fit_reducer, "explained_variance_ratio_")
                            else None
                        ),
                        "all_train_explained_variance": (
                            float(np.sum(all_reducer.explained_variance_ratio_))
                            if all_reducer is not None and hasattr(all_reducer, "explained_variance_ratio_")
                            else None
                        ),
                        "best_params": params,
                        **metrics,
                        "n_test": int(
                            np.isfinite(
                                pd.to_numeric(
                                    test[args.evaluation_target_column], errors="coerce"
                                )
                            ).sum()
                        ),
                    }
                )

    output = {
        "design": {
            "in_sample_years": 8,
            "fit_years": 6,
            "validation_years": 2,
            "test_years": 1,
            "train_target_column": args.train_target_column,
            "evaluation_target_column": args.evaluation_target_column,
            "threshold": 0.5,
            "max_input_chars": args.max_input_chars,
            "seed": args.seed,
        },
        "years": years,
        "results": results,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
