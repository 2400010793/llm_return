"""Cluster exact `收益` token embeddings in one leakage-safe rolling fold."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.prompt_token_mechanisms import SEEDS, select_stable_kmeans
from src.evaluation.artifacts import atomic_json


TARGETS = ("event_return_3d", "next_day_return")


def _positions(frame: pd.DataFrame, window: dict[str, object], target: str) -> dict[str, np.ndarray]:
    years = pd.to_datetime(frame["entry_date"], errors="coerce").dt.year.to_numpy()
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    return {
        "fit": np.flatnonzero(finite & np.isin(years, window["fit_years"])),
        "validation": np.flatnonzero(finite & np.isin(years, window["validation_years"])),
        "all_train": np.flatnonzero(finite & np.isin(years, window["all_train_years"])),
        "test": np.flatnonzero(finite & (years == int(window["test_year"]))),
    }


def _fit_pca(
    matrix: np.ndarray, positions: np.ndarray, *, components: int, sample_rows: int,
) -> PCA:
    selected = positions
    if len(selected) > sample_rows:
        selected = np.sort(np.random.default_rng(42).choice(selected, sample_rows, replace=False))
    values = np.asarray(matrix[selected], dtype=np.float32)
    count = min(components, values.shape[1], len(values) - 1)
    return PCA(
        n_components=count, svd_solver="randomized", random_state=42, iterated_power=3,
    ).fit(values)


def _probabilities_from_clusters(
    train_clusters: np.ndarray, train_returns: np.ndarray,
    test_clusters: np.ndarray, cluster_count: int,
) -> np.ndarray:
    global_rate = float(np.mean(train_returns > 0))
    rates = np.full(cluster_count, global_rate, dtype=np.float64)
    for cluster in range(cluster_count):
        selected = train_clusters == cluster
        if selected.any():
            rates[cluster] = float(np.mean(train_returns[selected] > 0))
    return rates[test_clusters]


def run(args: argparse.Namespace) -> dict[str, object]:
    shared_cache = args.mechanism_cache_root / args.model / args.prompt_length
    shared_manifest = json.loads((shared_cache / "manifest.json").read_text(encoding="utf-8"))
    matches = [
        row for row in shared_manifest["windows"]
        if int(row["test_year"]) == args.test_year
    ]
    if len(matches) != 1:
        raise ValueError(f"test year {args.test_year} is unavailable")
    window = matches[0]
    frame = pd.read_parquet(shared_cache / "rows.parquet")
    scopes = _positions(frame, window, args.target)
    span_cache = args.span_cache_root / args.model / args.prompt_length
    span_manifest = json.loads((span_cache / "manifest.json").read_text(encoding="utf-8"))
    if args.variant not in span_manifest["variants"]:
        raise ValueError(f"variant {args.variant} absent from span cache")
    matrix = np.load(span_cache / f"{args.variant}_return_span.npy", mmap_mode="r")
    returns = pd.to_numeric(frame[args.target], errors="coerce").to_numpy(dtype=float)

    fit_pca = _fit_pca(
        matrix, scopes["fit"], components=args.components, sample_rows=args.pca_sample_rows,
    )
    fit = fit_pca.transform(np.asarray(matrix[scopes["fit"]], dtype=np.float32))
    validation = fit_pca.transform(np.asarray(matrix[scopes["validation"]], dtype=np.float32))
    scaler = StandardScaler().fit(fit)
    fit = scaler.transform(fit).astype(np.float32)
    validation = scaler.transform(validation).astype(np.float32)
    selected_k, k_rows = select_stable_kmeans(
        fit, validation, k_values=range(2, 13), seeds=SEEDS,
        sample_size=args.silhouette_sample,
    )

    final_pca = _fit_pca(
        matrix, scopes["all_train"], components=args.components,
        sample_rows=args.pca_sample_rows,
    )
    all_train = final_pca.transform(np.asarray(matrix[scopes["all_train"]], dtype=np.float32))
    test = final_pca.transform(np.asarray(matrix[scopes["test"]], dtype=np.float32))
    final_scaler = StandardScaler().fit(all_train)
    all_train = final_scaler.transform(all_train).astype(np.float32)
    test = final_scaler.transform(test).astype(np.float32)
    cluster_model = MiniBatchKMeans(
        n_clusters=selected_k, random_state=42, batch_size=2048,
        n_init=3, max_iter=200,
    ).fit(all_train)
    train_clusters = cluster_model.predict(all_train)
    test_clusters = cluster_model.predict(test)
    train_returns = returns[scopes["all_train"]]
    test_returns = returns[scopes["test"]]
    probabilities = _probabilities_from_clusters(
        train_clusters, train_returns, test_clusters, selected_k,
    )
    majority_probability = float(np.mean(train_returns > 0))
    labels = test_returns > 0
    cluster_accuracy = float(np.mean((probabilities >= 0.5) == labels))
    majority_accuracy = float(np.mean((majority_probability >= 0.5) == labels))

    output = (
        args.output_root / args.model / args.prompt_length / args.variant
        / args.target / str(args.test_year)
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite span cluster fold: {output}")
    stage = output.with_name(f".{output.name}.partial.{os.getpid()}")
    stage.mkdir(parents=True)
    try:
        pd.DataFrame(k_rows).to_csv(stage / "k_selection.csv", index=False)
        prediction = frame.iloc[scopes["test"]][
            ["row_index", "entry_date", "stock_id", args.target]
        ].copy()
        prediction["label"] = labels.astype(np.int8)
        prediction["cluster"] = test_clusters
        prediction["cluster_probability"] = probabilities
        prediction["majority_probability"] = majority_probability
        prediction.to_parquet(stage / "predictions.parquet", index=False)
        composition = prediction.groupby("cluster").agg(
            rows=("label", "size"), positive_rate=("label", "mean"),
            mean_return=(args.target, "mean"), median_return=(args.target, "median"),
            stocks=("stock_id", "nunique"),
        ).reset_index()
        composition.to_csv(stage / "cluster_composition.csv", index=False)
        chosen = next(row for row in k_rows if int(row["k"]) == selected_k)
        report = {
            "format_version": "prompt_return_span_cluster_fold_v1",
            "model": args.model, "prompt_length": args.prompt_length,
            "variant": args.variant, "target": args.target,
            "test_year": args.test_year, "window": window,
            "phrase": span_manifest["phrase"], "tokens": span_manifest["tokens"],
            "selected_k": selected_k, "selected_k_diagnostics": chosen,
            "cluster_accuracy": cluster_accuracy,
            "majority_accuracy": majority_accuracy,
            "accuracy_delta": cluster_accuracy - majority_accuracy,
            "test_rows": int(len(labels)),
            "positive_rate_spread": float(
                composition["positive_rate"].max() - composition["positive_rate"].min()
            ),
            "mean_return_spread": float(
                composition["mean_return"].max() - composition["mean_return"].min()
            ),
        }
        atomic_json(stage / "metrics.json", report)
        (stage / "COMPLETED").write_text(
            "prompt_return_span_cluster_fold_v1\n", encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output)
        return {**report, "output": str(output)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--span-cache-root", type=Path, required=True)
    parser.add_argument("--mechanism-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--prompt-length", choices=("short", "long"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--test-year", type=int, choices=range(2018, 2027), required=True)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--pca-sample-rows", type=int, default=20000)
    parser.add_argument("--silhouette-sample", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
