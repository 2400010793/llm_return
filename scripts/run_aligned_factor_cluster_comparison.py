#!/usr/bin/env python3
"""Compare clustering methods on the new aligned prompt factors.

Runs one axis/representation over the 2018--2026 6+2+1 rolling protocol.
The token and body matrices are identical in row order; only the representation
is changed.  Validation selects all parameters, and test years are untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


COMPONENTS = (32, 64)
K_VALUES = (2, 4, 6, 8, 12)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
TEMPERATURES = (0.0, 0.5, 1.0, 2.0)
SHRINKAGES = (25.0, 100.0, 500.0)


def stock_days(frame: pd.DataFrame, prediction: np.ndarray, target: str) -> pd.DataFrame:
    x = frame[["stock_id", "entry_date", target]].copy()
    x["prediction"] = prediction
    x["entry_date"] = pd.to_datetime(x["entry_date"]).dt.normalize()
    return x.groupby(["stock_id", "entry_date"], as_index=False).agg(
        actual=(target, "mean"), prediction=("prediction", "mean")
    )


def rank_ic(days: pd.DataFrame) -> float:
    values = []
    for _, group in days.groupby("entry_date"):
        if len(group) < 10 or group["prediction"].nunique() < 2:
            continue
        value = spearmanr(group["prediction"], group["actual"]).statistic
        if np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def top20_ls(days: pd.DataFrame) -> float:
    values = []
    for _, group in days.groupby("entry_date"):
        if len(group) < 10:
            continue
        n = max(1, int(np.ceil(len(group) * 0.2)))
        ordered = group.sort_values("prediction")
        values.append(float(ordered.tail(n)["actual"].mean() - ordered.head(n)["actual"].mean()))
    return float(np.mean(values)) if values else float("nan")


def project(matrix: np.ndarray, idx: np.ndarray, pca: PCA, scaler: StandardScaler, components: int) -> np.ndarray:
    return scaler.transform(pca.transform(np.asarray(matrix[idx], dtype=np.float32)))[:, :components].astype(np.float32)


def fit_projection(matrix: np.ndarray, idx: np.ndarray, sample_rows: int) -> tuple[PCA, StandardScaler]:
    selected = idx
    if len(selected) > sample_rows:
        selected = np.sort(np.random.default_rng(42).choice(selected, sample_rows, replace=False))
    values = np.asarray(matrix[selected], dtype=np.float32)
    pca = PCA(n_components=min(64, values.shape[1], len(values) - 1), svd_solver="randomized", random_state=42, iterated_power=3).fit(values)
    scaler = StandardScaler().fit(pca.transform(values))
    return pca, scaler


def fit_kmeans(x: np.ndarray, k: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=3, max_iter=200).fit(x)


def shrunk_cluster_means(labels: np.ndarray, y: np.ndarray, k: int, shrinkage: float) -> np.ndarray:
    global_mean = float(np.mean(y))
    means = np.full(k, global_mean, dtype=float)
    for cluster in range(k):
        mask = labels == cluster
        if mask.any():
            means[cluster] = (float(y[mask].sum()) + shrinkage * global_mean) / (int(mask.sum()) + shrinkage)
    return means


def soft_prediction(model: MiniBatchKMeans, x: np.ndarray, means: np.ndarray, temperature: float) -> np.ndarray:
    distances = model.transform(x)
    if temperature == 0:
        return means[np.argmin(distances, axis=1)]
    squared = distances.astype(float) ** 2
    positive = squared[squared > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    logits = -squared / max(temperature * scale, 1e-12)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ means


def choose_linear(x_fit, y_fit, x_val, frame_val, target):
    rows = []
    for components in COMPONENTS:
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha).fit(x_fit[:, :components], y_fit)
            pred = model.predict(x_val[:, :components])
            rows.append((rank_ic(stock_days(frame_val, pred, target)), components, alpha))
    rows = [r for r in rows if np.isfinite(r[0])]
    return max(rows, key=lambda r: r[0]), rows


def choose_hard(x_fit, y_fit, x_val, frame_val, target):
    rows = []
    for components in COMPONENTS:
        for k in K_VALUES:
            km = fit_kmeans(x_fit[:, :components], k)
            z_fit = np.eye(k, dtype=np.float32)[km.predict(x_fit[:, :components])]
            z_val = np.eye(k, dtype=np.float32)[km.predict(x_val[:, :components])]
            for alpha in ALPHAS:
                model = Ridge(alpha=alpha).fit(np.hstack([x_fit[:, :components], z_fit]), y_fit)
                pred = model.predict(np.hstack([x_val[:, :components], z_val]))
                rows.append((rank_ic(stock_days(frame_val, pred, target)), components, k, alpha))
    rows = [r for r in rows if np.isfinite(r[0])]
    return max(rows, key=lambda r: r[0]), rows


def choose_soft(x_fit, y_fit, x_val, frame_val, target):
    rows = []
    for components in COMPONENTS:
        for k in K_VALUES:
            km = fit_kmeans(x_fit[:, :components], k)
            labels = km.predict(x_fit[:, :components])
            for shrinkage in SHRINKAGES:
                means = shrunk_cluster_means(labels, y_fit, k, shrinkage)
                for temperature in TEMPERATURES:
                    pred = soft_prediction(km, x_val[:, :components], means, temperature)
                    rows.append((rank_ic(stock_days(frame_val, pred, target)), components, k, shrinkage, temperature))
    rows = [r for r in rows if np.isfinite(r[0])]
    return max(rows, key=lambda r: r[0]), rows


def choose_umap(x_fit, y_fit, x_val, frame_val, target):
    reducer = umap.UMAP(n_components=8, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42, transform_seed=42, low_memory=True, n_jobs=1).fit(x_fit[:, :64])
    z_fit, z_val = reducer.transform(x_fit[:, :64]), reducer.transform(x_val[:, :64])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric="euclidean", prediction_data=True).fit(z_fit)
    labels_fit, strengths_fit = hdbscan.approximate_predict(clusterer, z_fit)
    labels_val, strengths_val = hdbscan.approximate_predict(clusterer, z_val)
    labels_fit = labels_fit.astype(int); labels_val = labels_val.astype(int)
    k = int(clusterer.labels_.max() + 2)
    labels_fit[labels_fit < 0] = k - 1; labels_val[labels_val < 0] = k - 1
    one_fit = np.eye(k, dtype=np.float32)[labels_fit]
    one_val = np.eye(k, dtype=np.float32)[labels_val]
    rows = []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(np.hstack([z_fit, one_fit, strengths_fit[:, None]]), y_fit)
        pred = model.predict(np.hstack([z_val, one_val, strengths_val[:, None]]))
        rows.append((rank_ic(stock_days(frame_val, pred, target)), alpha, reducer, clusterer))
    return max(rows, key=lambda r: r[0]), rows


def run(args: argparse.Namespace) -> None:
    panel = pd.read_parquet(args.panel)
    metadata = pd.read_parquet(args.metadata)
    matrix = np.load(args.matrix, mmap_mode="r")
    if not np.array_equal(panel["row_index"].to_numpy(), metadata["row_index"].to_numpy()):
        raise ValueError("panel and matrix metadata row_index mismatch")
    panel["entry_date"] = pd.to_datetime(panel["entry_date"]).dt.normalize()
    years = panel["entry_date"].dt.year.to_numpy()
    y_all = pd.to_numeric(panel[args.target], errors="coerce").to_numpy(float)
    results = []
    for test_year in range(args.first_year, args.last_year + 1):
        finite = np.isfinite(y_all)
        fit_idx = np.flatnonzero(finite & np.isin(years, np.arange(test_year - 8, test_year - 2)))
        val_idx = np.flatnonzero(finite & np.isin(years, np.arange(test_year - 2, test_year)))
        train_idx = np.flatnonzero(finite & np.isin(years, np.arange(test_year - 8, test_year)))
        test_idx = np.flatnonzero(finite & (years == test_year))
        pca, scaler = fit_projection(matrix, fit_idx, args.sample_rows)
        x_fit, x_val = project(matrix, fit_idx, pca, scaler, 64), project(matrix, val_idx, pca, scaler, 64)
        frame_val = panel.iloc[val_idx]
        selected = {}
        selected["pca_ridge"], _ = choose_linear(x_fit, y_all[fit_idx], x_val, frame_val, args.target)
        selected["hard_kmeans"], _ = choose_hard(x_fit, y_all[fit_idx], x_val, frame_val, args.target)
        selected["soft_kmeans"], _ = choose_soft(x_fit, y_all[fit_idx], x_val, frame_val, args.target)
        if not args.skip_umap:
            selected["umap_hdbscan"], _ = choose_umap(x_fit, y_all[fit_idx], x_val, frame_val, args.target)
        pca_all, scaler_all = fit_projection(matrix, train_idx, args.sample_rows)
        x_train, x_test = project(matrix, train_idx, pca_all, scaler_all, 64), project(matrix, test_idx, pca_all, scaler_all, 64)
        frame_test = panel.iloc[test_idx]
        # Refit each selected method using train+validation, with all parameter
        # choices frozen from validation.
        for method, choice in selected.items():
            if method == "pca_ridge":
                _, components, alpha = choice
                model = Ridge(alpha=alpha).fit(x_train[:, :components], y_all[train_idx])
                pred = model.predict(x_test[:, :components])
            elif method == "hard_kmeans":
                _, components, k, alpha = choice
                km = fit_kmeans(x_train[:, :components], k)
                z_train = np.eye(k, dtype=np.float32)[km.predict(x_train[:, :components])]
                z_test = np.eye(k, dtype=np.float32)[km.predict(x_test[:, :components])]
                model = Ridge(alpha=alpha).fit(np.hstack([x_train[:, :components], z_train]), y_all[train_idx])
                pred = model.predict(np.hstack([x_test[:, :components], z_test]))
            elif method == "soft_kmeans":
                _, components, k, shrinkage, temperature = choice
                km = fit_kmeans(x_train[:, :components], k)
                means = shrunk_cluster_means(km.predict(x_train[:, :components]), y_all[train_idx], k, shrinkage)
                pred = soft_prediction(km, x_test[:, :components], means, temperature)
            elif method == "umap_hdbscan":
                _, alpha, reducer, _ = choice
                z_train = reducer.fit_transform(x_train[:, :64])
                z_test = reducer.transform(x_test[:, :64])
                clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric="euclidean", prediction_data=True).fit(z_train)
                labels_train, strengths_train = hdbscan.approximate_predict(clusterer, z_train)
                labels_test, strengths_test = hdbscan.approximate_predict(clusterer, z_test)
                labels_train = labels_train.astype(int); labels_test = labels_test.astype(int)
                k = int(clusterer.labels_.max() + 2)
                labels_train[labels_train < 0] = k - 1; labels_test[labels_test < 0] = k - 1
                model = Ridge(alpha=alpha).fit(np.hstack([z_train, np.eye(k)[labels_train], strengths_train[:, None]]), y_all[train_idx])
                pred = model.predict(np.hstack([z_test, np.eye(k)[labels_test], strengths_test[:, None]]))
            days = stock_days(frame_test, pred, args.target)
            results.append({"axis": args.axis, "representation": args.representation, "target": args.target, "test_year": test_year, "method": method, "rankic": rank_ic(days), "top20_ls": top20_ls(days), "n_test": len(days), "validation_rankic": float(choice[0])})
        print(json.dumps({"axis": args.axis, "representation": args.representation, "target": args.target, "test_year": test_year}, ensure_ascii=False), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--axis", required=True)
    p.add_argument("--representation", choices=("token", "body"), required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--first-year", type=int, default=2018)
    p.add_argument("--last-year", type=int, default=2026)
    p.add_argument("--sample-rows", type=int, default=20000)
    p.add_argument("--skip-umap", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
