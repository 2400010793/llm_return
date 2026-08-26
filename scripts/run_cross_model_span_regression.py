"""Leakage-safe cross-model span regression and clustering.

Each model's span is scaled and reduced independently inside every rolling
window before concatenation.  ``--components 0`` runs the raw standardized
MiniBatchKMeans/Ridge control without PCA.  The runner is dataset agnostic:
the input roots only need the existing semantic span cache layout and a
mechanism cache containing ``rows.parquet`` and rolling windows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from run_prompt_span_return_regression import (
    ALPHAS,
    _cluster_predict,
    _ic,
    _valid,
    _stock_day,
)


def _matrix(cache_root: Path, model: str, semantic: str, variant: str) -> np.ndarray:
    path = cache_root / model / "short" / semantic / f"{variant}_{semantic}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model") != model or manifest.get("semantic_name") != semantic:
        raise ValueError(f"cache manifest mismatch: {path}")
    return np.load(path, mmap_mode="r")


def _project(matrix: np.ndarray, indices: np.ndarray, components: int, sample_rows: int):
    selected = indices
    if len(selected) > sample_rows:
        selected = np.sort(np.random.default_rng(42).choice(selected, sample_rows, replace=False))
    scaler = StandardScaler().fit(np.asarray(matrix[selected], dtype=np.float32))
    if components:
        count = min(components, matrix.shape[1], len(selected) - 1)
        reducer = PCA(n_components=count, svd_solver="randomized", random_state=42, iterated_power=3)
        reducer.fit(scaler.transform(np.asarray(matrix[selected], dtype=np.float32)))
    else:
        reducer = None

    def transform(pos: np.ndarray) -> np.ndarray:
        values = scaler.transform(np.asarray(matrix[pos], dtype=np.float32))
        return values if reducer is None else reducer.transform(values)

    return scaler, reducer, transform


def _fit_cluster(x: np.ndarray, k: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=k, random_state=42, batch_size=2048, n_init=3, max_iter=200,
    ).fit(x)


def run(args: argparse.Namespace) -> dict[str, object]:
    if len(args.models) != 2:
        raise ValueError("exactly two models are required for cross-model fusion")
    semantics = tuple(value.strip() for value in args.semantics.split(",") if value.strip())
    if not semantics:
        raise ValueError("at least one semantic span is required")
    mechanism = args.mechanism_cache_root / args.models[0] / "short"
    manifest = json.loads((mechanism / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(mechanism / "rows.parquet")
    matrices = {
        model: {
            semantic: _matrix(args.span_cache_root, model, semantic, args.variant)
            for semantic in semantics
        }
        for model in args.models
    }
    for model in args.models[1:]:
        other = pd.read_parquet(args.mechanism_cache_root / model / "short" / "rows.parquet")
        if not frame[["row_index", "stock_id", "entry_date"]].equals(other[["row_index", "stock_id", "entry_date"]]):
            raise ValueError(f"row alignment mismatch between {args.models[0]} and {model}")
    returns = pd.to_numeric(frame[args.target], errors="coerce").to_numpy(dtype=float)
    outputs = []
    for window in manifest["windows"]:
        test_year = int(window["test_year"])
        fit_idx = _valid(frame, window, args.target, "fit_years")
        val_idx = _valid(frame, window, args.target, "validation_years")
        all_idx = _valid(frame, window, args.target, "all_train_years")
        year = pd.to_datetime(frame["entry_date"], errors="coerce").dt.year.to_numpy()
        test_idx = np.flatnonzero(np.isfinite(returns) & (year == test_year))

        fit_parts, val_parts = [], []
        fit_artifacts = []
        for model in args.models:
            for semantic in semantics:
                scaler, reducer, transform = _project(
                    matrices[model][semantic], fit_idx, args.components, args.pca_sample_rows,
                )
                fit_parts.append(transform(fit_idx)); val_parts.append(transform(val_idx))
                fit_artifacts.append((model, semantic, scaler, reducer))
        x_fit, x_val = np.hstack(fit_parts).astype(np.float32), np.hstack(val_parts).astype(np.float32)
        y_fit = returns[fit_idx]

        best = (-np.inf, 0, ALPHAS[-1], None)
        k_values = args.k_values if args.cluster else (0,)
        for k in k_values:
            cluster = _fit_cluster(x_fit, int(k)) if k else None
            fit_x = x_fit if cluster is None else np.hstack((x_fit, np.eye(k, dtype=np.float32)[cluster.predict(x_fit)]))
            val_x = x_val if cluster is None else np.hstack((x_val, np.eye(k, dtype=np.float32)[cluster.predict(x_val)]))
            for alpha in ALPHAS:
                model = Ridge(alpha=alpha).fit(fit_x, y_fit)
                score = _ic(frame.iloc[val_idx], model.predict(val_x), args.target)
                if np.isfinite(score) and score > best[0]:
                    best = (score, int(k), float(alpha), cluster)

        final_parts_all, final_parts_test, final_artifacts = [], [], []
        for model in args.models:
            for semantic in semantics:
                scaler, reducer, transform = _project(
                    matrices[model][semantic], all_idx, args.components, args.pca_sample_rows,
                )
                final_parts_all.append(transform(all_idx)); final_parts_test.append(transform(test_idx))
                final_artifacts.append((model, semantic, scaler, reducer))
        x_all, x_test = np.hstack(final_parts_all).astype(np.float32), np.hstack(final_parts_test).astype(np.float32)
        final_cluster = _fit_cluster(x_all, best[1]) if best[1] else None
        if final_cluster is not None:
            x_all = np.hstack((x_all, np.eye(best[1], dtype=np.float32)[final_cluster.predict(x_all)]))
            x_test = np.hstack((x_test, np.eye(best[1], dtype=np.float32)[final_cluster.predict(x_test)]))
        final_model = Ridge(alpha=best[2]).fit(x_all, returns[all_idx])
        prediction = final_model.predict(x_test)
        daily = _stock_day(frame.iloc[test_idx], prediction, args.target)
        out = args.output_root / args.variant / args.target / str(test_year)
        out.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(out / "stock_day_predictions.parquet", index=False)
        pd.DataFrame({"row_index": frame.iloc[test_idx]["row_index"].to_numpy(), "prediction": prediction}).to_parquet(out / "news_predictions.parquet", index=False)
        if final_cluster is not None:
            labels = final_cluster.predict(np.hstack(final_parts_test).astype(np.float32))
            pd.DataFrame({"cluster": labels, args.target: returns[test_idx]}).groupby("cluster").agg(
                rows=(args.target, "size"), mean_return=(args.target, "mean"),
                median_return=(args.target, "median"), std_return=(args.target, "std"),
            ).reset_index().to_csv(out / "cluster_return_composition.csv", index=False)
        joblib.dump({"artifacts": final_artifacts, "models": args.models, "semantics": semantics}, out / "preprocessor.joblib")
        joblib.dump(final_cluster, out / "cluster.joblib")
        joblib.dump(final_model, out / "ridge.joblib")
        report = {
            "format_version": "cross_model_span_regression_v1", "models": args.models,
            "semantics": semantics, "variant": args.variant, "target": args.target,
            "test_year": test_year, "components_per_model": args.components,
            "cluster": args.cluster, "selected_k": best[1], "alpha": best[2],
            "validation_ic": float(best[0]), "test_ic": _ic(frame.iloc[test_idx], prediction, args.target),
            "rows": {"fit": len(fit_idx), "validation": len(val_idx), "all_train": len(all_idx), "test": len(test_idx)},
        }
        (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "COMPLETED").write_text("cross_model_span_regression_v1\n", encoding="utf-8")
        outputs.append(report)
    summary = {"format_version": "cross_model_span_regression_summary_v1", "models": args.models, "semantics": semantics, "variant": args.variant, "target": args.target, "results": outputs}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"summary_{args.variant}_{args.target}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--span-cache-root", type=Path, required=True)
    p.add_argument("--mechanism-cache-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--models", nargs=2, default=("roberta", "bge_m3"))
    p.add_argument("--semantics", default="return_span")
    p.add_argument("--variant", choices=("short", "masked_short"), required=True)
    p.add_argument("--target", choices=("next_day_return", "event_return_3d"), required=True)
    p.add_argument("--components", type=int, default=64)
    p.add_argument("--pca-sample-rows", type=int, default=20000)
    p.add_argument("--cluster", action="store_true")
    p.add_argument("--k-values", type=int, nargs="+", default=(2, 4, 6, 8, 12))
    args = p.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
