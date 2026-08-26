#!/usr/bin/env python3
"""Run one fixed-parameter PCA token/body fold across models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from itertools import combinations
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler


SEEDS = (17, 29, 42, 71, 113)
FIXED_COMPONENTS = 32
FIXED_ALPHA = 100.0
FIXED_K = 6
FIXED_SHRINKAGE = 500.0
FIXED_TEMPERATURE = 0.5


def stock_days(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    result = frame[["stock_id", "entry_date", "next_day_return"]].copy()
    result["prediction"] = prediction
    return result.groupby(["stock_id", "entry_date"], as_index=False).agg(
        actual_return=("next_day_return", "mean"), prediction=("prediction", "mean"),
        news=("prediction", "size"),
    )


def daily_metrics(days: pd.DataFrame) -> dict[str, float]:
    ics, top, bottom, spreads = [], [], [], []
    for _, group in days.groupby("entry_date", sort=False):
        group = group.dropna(subset=["actual_return", "prediction"])
        if len(group) < 5 or group["prediction"].nunique() < 2:
            continue
        ic = spearmanr(group["prediction"], group["actual_return"]).statistic
        if np.isfinite(ic):
            ics.append(float(ic))
        count = max(1, int(np.ceil(len(group) * 0.2)))
        ordered = group.sort_values(["prediction", "stock_id"], kind="mergesort")
        low = float(ordered.head(count)["actual_return"].mean())
        high = float(ordered.tail(count)["actual_return"].mean())
        top.append(high); bottom.append(low); spreads.append(high - low)
    values = np.asarray(ics, dtype=float)
    return {
        "rank_ic": float(values.mean()) if len(values) else float("nan"),
        "rank_ic_ir": float(values.mean() / values.std(ddof=1)) if len(values) > 1 and values.std(ddof=1) else float("nan"),
        "rank_ic_days": len(values),
        "top20_return": float(np.mean(top)) if top else float("nan"),
        "bottom20_return": float(np.mean(bottom)) if bottom else float("nan"),
        "top20_long_short": float(np.mean(spreads)) if spreads else float("nan"),
        "direction_accuracy": float(np.mean(np.sign(days.prediction) == np.sign(days.actual_return))),
        "oos_r2": float(r2_score(days.actual_return, days.prediction)) if len(days) > 1 else float("nan"),
    }


def projection(matrix: np.ndarray, fit: np.ndarray, predict: np.ndarray, components: int | None):
    x_fit = np.asarray(matrix[fit], dtype=np.float32)
    x_predict = np.asarray(matrix[predict], dtype=np.float32)
    reducer = None
    if components is not None:
        reducer = PCA(
            n_components=min(components, x_fit.shape[1], len(x_fit) - 1),
            svd_solver="randomized", random_state=42, iterated_power=3,
        )
        x_fit = reducer.fit_transform(x_fit)
        x_predict = reducer.transform(x_predict)
    scaler = StandardScaler().fit(x_fit)
    return scaler.transform(x_fit).astype(np.float32), scaler.transform(x_predict).astype(np.float32), reducer, scaler


def kmeans_ensemble(x: np.ndarray, k: int):
    return [
        MiniBatchKMeans(
            n_clusters=k, random_state=seed, batch_size=2048, n_init=3, max_iter=200,
        ).fit(x) for seed in SEEDS
    ]


def one_hot(models, x: np.ndarray) -> np.ndarray:
    return np.hstack([np.eye(model.n_clusters, dtype=np.float32)[model.predict(x)] for model in models])


def cluster_stability(models, x: np.ndarray) -> float:
    labels = [model.predict(x) for model in models]
    values = [adjusted_rand_score(labels[a], labels[b]) for a, b in combinations(range(len(labels)), 2)]
    return float(np.mean(values))


def shrunk_means(labels, values, k: int, shrinkage: float) -> np.ndarray:
    mean = float(np.mean(values))
    result = np.full(k, mean, dtype=float)
    for cluster in range(k):
        mask = labels == cluster
        if mask.any():
            result[cluster] = (float(values[mask].sum()) + shrinkage * mean) / (int(mask.sum()) + shrinkage)
    return result


def soft_values(model, x, means, temperature: float) -> np.ndarray:
    distance = np.square(model.transform(x).astype(float))
    if temperature == 0:
        return means[np.argmin(distance, axis=1)]
    positive = distance[distance > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    logits = -distance / max(temperature * scale, 1e-12)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits); weights /= weights.sum(axis=1, keepdims=True)
    return weights @ means


def hdbscan_features(x_fit: np.ndarray, x_predict: np.ndarray):
    reducer = umap.UMAP(
        n_components=8, n_neighbors=30, min_dist=0.1, metric="cosine",
        random_state=42, transform_seed=42, low_memory=True, n_jobs=1,
    ).fit(x_fit)
    z_fit, z_predict = reducer.transform(x_fit), reducer.transform(x_predict)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50, min_samples=10, metric="euclidean", prediction_data=True,
    ).fit(z_fit)
    fit_labels, fit_strength = hdbscan.approximate_predict(clusterer, z_fit)
    pred_labels, pred_strength = hdbscan.approximate_predict(clusterer, z_predict)
    existing = sorted(label for label in np.unique(fit_labels) if label >= 0)
    mapping = {label: index for index, label in enumerate(existing)}
    noise = len(existing)
    encode = lambda labels: np.asarray([mapping.get(int(label), noise) for label in labels], dtype=int)
    fit_encoded, pred_encoded = encode(fit_labels), encode(pred_labels)
    k = noise + 1
    x_fit_out = np.hstack([z_fit, np.eye(k, dtype=np.float32)[fit_encoded], fit_strength[:, None]])
    x_predict_out = np.hstack([z_predict, np.eye(k, dtype=np.float32)[pred_encoded], pred_strength[:, None]])
    return x_fit_out, x_predict_out, reducer, clusterer, int(k), float(np.mean(pred_labels < 0))


def fixed_predictions(matrix, train, predict, y):
    """Fit the preregistered PCA32 models without using a validation search."""
    x_train, x_predict, _, _ = projection(
        matrix, train, predict, FIXED_COMPONENTS,
    )
    predictions = {
        "ridge_pca32": Ridge(alpha=FIXED_ALPHA).fit(
            x_train, y[train],
        ).predict(x_predict),
    }
    models = kmeans_ensemble(x_train, FIXED_K)
    train_one_hot, predict_one_hot = one_hot(models, x_train), one_hot(models, x_predict)
    predictions["hard_kmeans"] = Ridge(alpha=FIXED_ALPHA).fit(
        np.hstack([x_train, train_one_hot]), y[train],
    ).predict(np.hstack([x_predict, predict_one_hot]))
    predictions["soft_kmeans"] = np.mean([
        soft_values(
            model, x_predict,
            shrunk_means(model.predict(x_train), y[train], FIXED_K, FIXED_SHRINKAGE),
            FIXED_TEMPERATURE,
        )
        for model in models
    ], axis=0)
    htrain, hpredict, _, _, hclusters, noise = hdbscan_features(x_train, x_predict)
    predictions["umap_hdbscan"] = Ridge(alpha=FIXED_ALPHA).fit(
        htrain, y[train],
    ).predict(hpredict)
    sample = np.random.default_rng(42).choice(
        len(x_predict), min(3000, len(x_predict)), replace=False,
    )
    labels = models[0].predict(x_predict)
    diagnostics = {
        "hard_kmeans_ari": cluster_stability(models, x_predict),
        "hard_kmeans_silhouette": float(silhouette_score(x_predict[sample], labels[sample]))
        if len(np.unique(labels[sample])) > 1 else float("nan"),
        "hdbscan_clusters_including_noise": hclusters,
        "hdbscan_noise_fraction": noise,
    }
    return predictions, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--representation", choices=("token", "body"), required=True)
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--history-years", type=int, required=True)
    parser.add_argument("--validation-years", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "COMPLETED").is_file():
        print(json.dumps({"resumed": True, "output": str(args.output)})); return

    panel = pd.read_parquet(args.panel)
    metadata = pd.read_parquet(args.metadata)
    matrix = np.load(args.matrix, mmap_mode="r")
    if len(panel) != len(matrix) or not np.array_equal(panel.row_index.to_numpy(), metadata.row_index.to_numpy()):
        raise ValueError("panel/matrix row_index mismatch")
    panel["entry_date"] = pd.to_datetime(panel["entry_date"]).dt.normalize()
    years = panel.entry_date.dt.year.to_numpy()
    y = pd.to_numeric(panel.next_day_return, errors="coerce").to_numpy(float)
    finite = np.isfinite(y)
    train_years = list(range(args.test_year - args.history_years, args.test_year))
    train = np.flatnonzero(finite & np.isin(years, train_years))
    test = np.flatnonzero(finite & (years == args.test_year))
    if min(map(len, (train, test))) == 0:
        raise ValueError(f"empty split for {args.test_year}")
    test_predictions, diagnostics = fixed_predictions(matrix, train, test, y)
    validation_predictions = {}
    validation_years = []
    val = np.asarray([], dtype=int)
    fit = np.asarray([], dtype=int)
    if args.validation_years:
        if args.validation_years >= args.history_years:
            raise ValueError("validation-years must be smaller than history-years")
        validation_years = train_years[-args.validation_years:]
        fit_years = train_years[:-args.validation_years]
        fit = np.flatnonzero(finite & np.isin(years, fit_years))
        val = np.flatnonzero(finite & np.isin(years, validation_years))
        validation_predictions, _ = fixed_predictions(matrix, fit, val, y)

    stage = args.output.with_name(f".{args.output.name}.partial.{os.getpid()}")
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    validation_frames, test_frames, metric_rows = [], [], []
    for method, prediction in validation_predictions.items():
        days = stock_days(panel.iloc[val], prediction); days["method"] = method
        validation_frames.append(days)
    for method, prediction in test_predictions.items():
        days = stock_days(panel.iloc[test], prediction); days["method"] = method
        test_frames.append(days)
        metric_rows.append({"model": args.model, "prompt": args.prompt, "representation": args.representation,
                            "test_year": args.test_year, "method": method, "stock_days": len(days), **daily_metrics(days)})
    if validation_frames:
        pd.concat(validation_frames, ignore_index=True).to_parquet(
            stage / "validation_stock_day_predictions.parquet", index=False,
        )
    pd.concat(test_frames, ignore_index=True).to_parquet(stage / "test_stock_day_predictions.parquet", index=False)
    pd.DataFrame(metric_rows).to_csv(stage / "metrics.csv", index=False)
    fixed_parameters = {
        "pca_components": FIXED_COMPONENTS, "ridge_alpha": FIXED_ALPHA,
        "kmeans_k": FIXED_K, "kmeans_seeds": list(SEEDS),
        "soft_shrinkage": FIXED_SHRINKAGE,
        "soft_temperature": FIXED_TEMPERATURE,
        "umap": {"n_components": 8, "n_neighbors": 30, "min_dist": 0.1, "metric": "cosine"},
        "hdbscan": {"min_cluster_size": 50, "min_samples": 10},
    }
    (stage / "fixed_parameters.json").write_text(
        json.dumps(fixed_parameters, indent=2) + "\n", encoding="utf-8",
    )
    (stage / "cluster_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=float) + "\n", encoding="utf-8",
    )
    manifest = {"format_version": "three_model_prompt_fair_fold_pca_v2", "model": args.model, "prompt": args.prompt,
                "representation": args.representation, "test_year": args.test_year,
                "fit_years": train_years[:-args.validation_years] if args.validation_years else [],
                "validation_years": validation_years,
                "train_years": train_years, "protocol": f"{args.history_years}+1",
                "representations": ["pca32"], "raw_regression": False,
                "hyperparameter_search": False, "fixed_parameters": fixed_parameters,
                "rows": {"fit": len(fit), "validation": len(val), "train": len(train), "test": len(test)}}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (stage / "COMPLETED").write_text("three_model_prompt_fair_fold_pca_v2\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    stage.replace(args.output)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
