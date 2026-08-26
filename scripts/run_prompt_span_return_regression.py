"""Leakage-safe rolling Ridge regression on one contextualized prompt span.

The span cache is reduced with PCA fit on the historical window only.  A
MiniBatchKMeans one-hot is optionally appended; k and alpha are selected on
the two validation years by stock-day rank IC.  PCA/scaler/cluster artifacts
are persisted for every test year so the result is auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.prompt_token_mechanisms import SEEDS, select_stable_kmeans


ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
KNN_CONFIGS = ((5, "uniform"), (15, "uniform"), (30, "uniform"),
               (60, "uniform"), (15, "distance"), (30, "distance"))


def _valid(frame: pd.DataFrame, window: dict[str, object], target: str, key: str) -> np.ndarray:
    years = pd.to_datetime(frame["entry_date"], errors="coerce").dt.year.to_numpy()
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y)
    return np.flatnonzero(finite & np.isin(years, window[key]))


def _stock_day(frame: pd.DataFrame, prediction: np.ndarray, target: str) -> pd.DataFrame:
    data = frame[["stock_id", "entry_date", target]].copy()
    data["prediction"] = prediction
    return data.groupby(["stock_id", "entry_date"], as_index=False).agg(
        prediction=("prediction", "mean"), actual=(target, "mean"),
    )


def _ic(frame: pd.DataFrame, prediction: np.ndarray, target: str) -> float:
    daily = _stock_day(frame, prediction, target)
    values = []
    for _, group in daily.groupby("entry_date"):
        if len(group) >= 5 and group["prediction"].nunique() > 1 and group["actual"].nunique() > 1:
            values.append(group["prediction"].corr(group["actual"], method="spearman"))
    return float(np.nanmean(values)) if values else float("nan")


def _pca(matrix: np.ndarray, indices: np.ndarray, components: int, sample: int) -> PCA:
    chosen = indices
    if len(chosen) > sample:
        chosen = np.sort(np.random.default_rng(42).choice(chosen, sample, replace=False))
    count = min(components, matrix.shape[1], len(chosen) - 1)
    return PCA(n_components=count, svd_solver="randomized", random_state=42, iterated_power=3).fit(
        np.asarray(matrix[chosen], dtype=np.float32)
    )


def _cluster_fit(method: str, x: np.ndarray, k: int, hierarchy_sample: int = 4000):
    if method == "kmeans":
        return MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=3, max_iter=200).fit(x)
    if method == "gmm":
        return GaussianMixture(n_components=k, covariance_type="diag", random_state=42, n_init=2, max_iter=200).fit(x)
    # Full agglomerative clustering is O(n^2). Fit on a deterministic
    # historical prototype sample, then assign every row by nearest centroid.
    sample_size = min(int(hierarchy_sample), len(x))
    sample_idx = np.sort(np.random.default_rng(42).choice(len(x), sample_size, replace=False))
    sample_model = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(x[sample_idx])
    centroids = np.vstack([
        x[sample_idx][sample_model.labels_ == i].mean(axis=0)
        for i in range(k)
    ])
    return {
        "model": "agglomerative_prototype", "centroids": centroids,
        "n_clusters": k, "sample_size": sample_size,
    }


def _cluster_predict(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        distances = ((x[:, None, :] - model["centroids"][None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)
    return model.predict(x)


def _cluster_count(model) -> int:
    if isinstance(model, dict):
        return int(model["n_clusters"])
    return int(getattr(model, "n_clusters", getattr(model, "n_components", 0)))


def _features(model, x: np.ndarray, append_cluster: bool) -> np.ndarray:
    if not append_cluster:
        return x
    labels = _cluster_predict(model, x)
    count = _cluster_count(model)
    one_hot = np.zeros((len(x), count), dtype=np.float32)
    one_hot[np.arange(len(x)), labels] = 1.0
    return np.hstack((x, one_hot))


def run(args: argparse.Namespace) -> dict[str, object]:
    semantic_names = tuple(value.strip() for value in args.semantic_name.split(",") if value.strip())
    if not semantic_names:
        raise ValueError("at least one semantic span is required")
    caches = [args.span_cache_root / args.model / args.prompt_length / name for name in semantic_names]
    manifests = [json.loads((cache / "manifest.json").read_text(encoding="utf-8")) for cache in caches]
    matrices = [np.load(cache / f"{args.variant}_{name}.npy", mmap_mode="r") for cache, name in zip(caches, semantic_names)]
    shared = args.mechanism_cache_root / args.model / args.prompt_length
    windows = json.loads((shared / "manifest.json").read_text(encoding="utf-8"))["windows"]
    frame = pd.read_parquet(shared / "rows.parquet")
    returns = pd.to_numeric(frame[args.target], errors="coerce").to_numpy(dtype=float)
    outputs = []
    for window in windows:
        test_year = int(window["test_year"])
        fit_idx = _valid(frame, window, args.target, "fit_years")
        val_idx = _valid(frame, window, args.target, "validation_years")
        all_idx = _valid(frame, window, args.target, "all_train_years")
        test_idx = np.flatnonzero(pd.to_datetime(frame["entry_date"]).dt.year.to_numpy() == test_year)
        pcas, scalers = [], []
        fit_parts, val_parts = [], []
        for matrix in matrices:
            pca = _pca(matrix, fit_idx, args.components, args.pca_sample_rows)
            scaler = StandardScaler().fit(pca.transform(np.asarray(matrix[fit_idx], dtype=np.float32)))
            pcas.append(pca); scalers.append(scaler)
            fit_parts.append(scaler.transform(pca.transform(np.asarray(matrix[fit_idx], dtype=np.float32))))
            val_parts.append(scaler.transform(pca.transform(np.asarray(matrix[val_idx], dtype=np.float32))))
        x_fit = np.hstack(fit_parts).astype(np.float32)
        x_val = np.hstack(val_parts).astype(np.float32)
        y_fit, y_val = returns[fit_idx], returns[val_idx]

        if args.cluster_augmented:
            selected_k, k_audit = select_stable_kmeans(
                x_fit, x_val, k_values=range(2, 13), seeds=SEEDS, sample_size=args.silhouette_sample,
            )
            candidate_ks = sorted(set([selected_k] + [int(row["k"]) for row in k_audit]))
        else:
            selected_k, k_audit, candidate_ks = 0, [], [0]
        best = (-np.inf, selected_k, ALPHAS[-1], None)
        for k in candidate_ks:
            if not args.cluster_augmented:
                configs = (("ridge", alpha, None) for alpha in ALPHAS) if args.predictor == "ridge" else (("knn", n, w) for n, w in KNN_CONFIGS)
                for predictor, parameter, weight in configs:
                    model = Ridge(alpha=parameter).fit(x_fit, y_fit) if predictor == "ridge" else KNeighborsRegressor(n_neighbors=parameter, weights=weight, n_jobs=-1).fit(x_fit, y_fit)
                    score = _ic(frame.iloc[val_idx], model.predict(x_val), args.target)
                    if np.isfinite(score) and score > best[0]:
                        best = (score, 0, parameter, None, predictor, weight)
                break
            cluster = _cluster_fit(args.cluster_method, x_fit, k, args.hierarchy_sample)
            configs = (("ridge", alpha, None) for alpha in ALPHAS) if args.predictor == "ridge" else (("knn", n, w) for n, w in KNN_CONFIGS)
            for predictor, parameter, weight in configs:
                model = Ridge(alpha=parameter).fit(_features(cluster, x_fit, True), y_fit) if predictor == "ridge" else KNeighborsRegressor(n_neighbors=parameter, weights=weight, n_jobs=-1).fit(_features(cluster, x_fit, True), y_fit)
                score = _ic(frame.iloc[val_idx], model.predict(_features(cluster, x_val, args.cluster_augmented)), args.target)
                if np.isfinite(score) and score > best[0]:
                    best = (score, k, parameter, cluster, predictor, weight)

        final_pcas, final_scalers = [], []
        final_parts_all, final_parts_test = [], []
        for matrix in matrices:
            final_pca = _pca(matrix, all_idx, args.components, args.pca_sample_rows)
            final_scaler = StandardScaler().fit(final_pca.transform(np.asarray(matrix[all_idx], dtype=np.float32)))
            final_pcas.append(final_pca); final_scalers.append(final_scaler)
            final_parts_all.append(final_scaler.transform(final_pca.transform(np.asarray(matrix[all_idx], dtype=np.float32))))
            final_parts_test.append(final_scaler.transform(final_pca.transform(np.asarray(matrix[test_idx], dtype=np.float32))))
        x_all = np.hstack(final_parts_all).astype(np.float32)
        x_test = np.hstack(final_parts_test).astype(np.float32)
        final_cluster = _cluster_fit(args.cluster_method, x_all, int(best[1]), args.hierarchy_sample) if args.cluster_augmented else None
        train_features = _features(final_cluster, x_all, args.cluster_augmented)
        test_features = _features(final_cluster, x_test, args.cluster_augmented)
        final_model = Ridge(alpha=float(best[2])).fit(train_features, returns[all_idx]) if best[4] == "ridge" else KNeighborsRegressor(n_neighbors=int(best[2]), weights=best[5], n_jobs=-1).fit(train_features, returns[all_idx])
        prediction = final_model.predict(test_features)
        daily = _stock_day(frame.iloc[test_idx], prediction, args.target)
        cluster_labels = _cluster_predict(final_cluster, x_test) if args.cluster_augmented else np.zeros(len(test_idx), dtype=int)
        composition = pd.DataFrame({"cluster": cluster_labels, "actual": returns[test_idx]}).groupby("cluster").agg(
            rows=("actual", "size"), mean_return=("actual", "mean"), median_return=("actual", "median"),
        ).reset_index()
        out = args.output_root / args.model / args.prompt_length / args.semantic_name / args.variant / args.target / str(test_year)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pcas": final_pcas, "scalers": final_scalers, "semantic_names": semantic_names}, out / "pca.joblib")
        joblib.dump(final_cluster, out / "cluster.joblib")
        joblib.dump(final_model, out / f"{args.predictor}.joblib")
        daily.to_parquet(out / "stock_day_predictions.parquet", index=False)
        composition.to_csv(out / "cluster_return_composition.csv", index=False)
        outputs.append({
            "test_year": test_year, "fit_years": window["fit_years"], "validation_years": window["validation_years"],
            "semantic_name": args.semantic_name, "phrases": [m["phrase"] for m in manifests], "model": args.model,
            "prompt_length": args.prompt_length, "variant": args.variant, "target": args.target,
            "components": int(sum(p.n_components_ for p in final_pcas)), "selected_k": int(best[1]), "predictor": best[4], "parameter": float(best[2]), "distance_weight": best[5], "cluster_method": args.cluster_method, "hierarchy_sample": args.hierarchy_sample,
            "validation_ic": float(best[0]), "test_ic": _ic(frame.iloc[test_idx], prediction, args.target),
            "test_rows": int(len(test_idx)), "cluster_return_spread": float(composition["mean_return"].max() - composition["mean_return"].min()),
            "output": str(out),
        })
    summary = {"format_version": "prompt_semantic_span_return_regression_v2", "semantic_names": semantic_names, "cache_manifests": manifests, "results": outputs}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / args.model / args.prompt_length / args.semantic_name / args.variant / f"{args.target}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--span-cache-root", type=Path, required=True)
    parser.add_argument("--mechanism-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--prompt-length", choices=("short", "long"), default="short")
    parser.add_argument("--semantic-name", required=True, help="One name or comma-separated names, e.g. future_span,return_span")
    parser.add_argument("--variant", choices=("short", "masked_short"), required=True)
    parser.add_argument("--target", choices=("next_day_return", "event_return_3d"), default="next_day_return")
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--pca-sample-rows", type=int, default=20000)
    parser.add_argument("--silhouette-sample", type=int, default=2000)
    parser.add_argument("--cluster-augmented", action="store_true")
    parser.add_argument("--cluster-method", choices=("kmeans", "gmm", "agglomerative"), default="kmeans")
    parser.add_argument("--hierarchy-sample", type=int, default=4000)
    parser.add_argument("--predictor", choices=("ridge", "knn"), default="ridge")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
