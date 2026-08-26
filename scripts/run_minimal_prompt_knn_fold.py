"""Analyze one minimal-prompt rolling fold and test cluster-aware KNN."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.prompt_token_mechanisms import (
    SEEDS,
    aggregate_moments,
    cluster_token_responses,
    select_stable_kmeans,
    token_metrics_from_moments,
)
from src.evaluation.artifacts import atomic_json


TARGETS = ("event_return_3d", "next_day_return")
SHORT_GROUPS = ("analysis", "target_stock", "future_return")
LONG_GROUPS = (
    "performance_cashflow",
    "orders_investment",
    "financing_equity",
    "regulation_litigation",
    "operating_risk",
    "industry_change",
    "company_governance",
)
K_VALUES = (11, 31, 101)
KNN_METRICS = ("euclidean", "cosine")
KNN_WEIGHTS = ("uniform", "distance")
APPROXIMATE_KNN_MIN_ROWS = 50_000


def _scope_positions(frame: pd.DataFrame, window: dict[str, object], target: str) -> dict[str, np.ndarray]:
    years = pd.to_datetime(frame["entry_date"], errors="coerce").dt.year.to_numpy()
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    return {
        "fit": np.flatnonzero(finite & np.isin(years, window["fit_years"])),
        "validation": np.flatnonzero(finite & np.isin(years, window["validation_years"])),
        "all_train": np.flatnonzero(finite & np.isin(years, window["all_train_years"])),
        "test": np.flatnonzero(finite & (years == int(window["test_year"]))),
    }


def _labels(frame: pd.DataFrame, target: str, positions: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)[positions]
    return (values > 0).astype(np.int8)


def _accuracy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) >= 0.5) == np.asarray(y_true)))


def _fit_group_pca(
    groups: np.ndarray,
    positions: np.ndarray,
    group_indexes: np.ndarray,
    *,
    components: int,
    sample_vectors: int,
    seed: int,
) -> PCA:
    rng = np.random.default_rng(seed)
    maximum_rows = max(1, sample_vectors // len(group_indexes))
    selected = positions
    if len(selected) > maximum_rows:
        selected = np.sort(rng.choice(selected, maximum_rows, replace=False))
    if groups.ndim == 2:
        if group_indexes.tolist() != [0]:
            raise ValueError("a two-dimensional span cache must contain exactly one group")
        fit = np.asarray(groups[selected], dtype=np.float32)
    else:
        fit = np.concatenate(
            [np.asarray(groups[selected, index, :], dtype=np.float32) for index in group_indexes],
            axis=0,
        )
    count = min(int(components), fit.shape[1], len(fit) - 1)
    if count < 1:
        raise ValueError("group PCA requires at least two vectors")
    return PCA(
        n_components=count, svd_solver="randomized", random_state=seed,
        iterated_power=3,
    ).fit(fit)


def _group_signature(
    groups: np.ndarray,
    positions: np.ndarray,
    group_indexes: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    if groups.ndim == 2:
        if group_indexes.tolist() != [0]:
            raise ValueError("a two-dimensional span cache must contain exactly one group")
        return pca.transform(
            np.asarray(groups[positions], dtype=np.float32)
        ).astype(np.float32, copy=False)
    projected = [
        pca.transform(np.asarray(groups[positions, index, :], dtype=np.float32))
        for index in group_indexes
    ]
    return np.concatenate(projected, axis=1).astype(np.float32, copy=False)


def _neighbor_probabilities(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    *,
    metric: str,
    neighbors: int,
    weights: str,
    jobs: int,
) -> np.ndarray:
    count = min(int(neighbors), len(x_train))
    distances, indexes, _ = _kneighbors(
        x_train, x_query, metric=metric, neighbors=count, jobs=jobs,
    )
    neighbor_labels = y_train[indexes].astype(np.float64)
    if weights == "uniform":
        return neighbor_labels.mean(axis=1)
    inverse = 1.0 / np.maximum(distances, 1e-12)
    exact = distances <= 1e-12
    has_exact = exact.any(axis=1)
    probabilities = np.sum(inverse * neighbor_labels, axis=1) / np.sum(inverse, axis=1)
    if has_exact.any():
        probabilities[has_exact] = (
            np.sum(exact[has_exact] * neighbor_labels[has_exact], axis=1)
            / np.sum(exact[has_exact], axis=1)
        )
    return probabilities


def _kneighbors(
    x_train: np.ndarray,
    x_query: np.ndarray,
    *,
    metric: str,
    neighbors: int,
    jobs: int,
    approximate_min_rows: int = APPROXIMATE_KNN_MIN_ROWS,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Use exact KNN for small samples and CatBoost HNSW at panel scale."""
    count = min(int(neighbors), len(x_train))
    if len(x_train) < approximate_min_rows:
        model = NearestNeighbors(
            n_neighbors=count, metric=metric, n_jobs=jobs,
        ).fit(x_train)
        distances, indexes = model.kneighbors(x_query, return_distance=True)
        return distances, indexes, "sklearn_exact"

    from catboost.hnsw import EDistance, HnswEstimator

    train = np.ascontiguousarray(x_train, dtype=np.float32)
    query = np.ascontiguousarray(x_query, dtype=np.float32)
    if metric == "cosine":
        train_norm = np.linalg.norm(train, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        train = np.divide(train, train_norm, out=np.zeros_like(train), where=train_norm > 0)
        query = np.divide(query, query_norm, out=np.zeros_like(query), where=query_norm > 0)
        distance = EDistance.DotProduct
    elif metric == "euclidean":
        distance = EDistance.L2Sqr
    else:
        raise ValueError(f"unsupported KNN metric: {metric}")
    search_size = max(300, count * 3)
    model = HnswEstimator(
        n_neighbors=count, distance=distance, max_neighbors=32,
        search_neighborhood_size=search_size,
        num_exact_candidates=max(100, count * 2),
        batch_size=10_000, upper_level_batch_size=40_000,
    ).fit(train, num_threads=jobs, report_progress=False)
    raw, indexes = model.kneighbors(
        query, n_neighbors=count, return_distance=True,
        search_neighborhood_size=search_size,
    )
    if metric == "cosine":
        distances = np.maximum(0.0, 1.0 - raw)
    else:
        distances = np.sqrt(np.maximum(0.0, raw))
    indexes = indexes.astype(np.int64, copy=False)
    distances, indexes, repaired_rows = _repair_invalid_neighbors(
        train, query, distances, indexes,
        metric=metric, neighbors=count, jobs=jobs,
    )
    if repaired_rows:
        warnings.warn(
            "CatBoost HNSW returned invalid results for "
            f"{repaired_rows}/{len(query)} query rows; repaired with exact KNN",
            RuntimeWarning,
            stacklevel=2,
        )
    return distances, indexes, "catboost_hnsw"


def _repair_invalid_neighbors(
    x_train: np.ndarray,
    x_query: np.ndarray,
    distances: np.ndarray,
    indexes: np.ndarray,
    *,
    metric: str,
    neighbors: int,
    jobs: int,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Replace invalid approximate-neighbor rows with exact KNN results."""
    invalid = (
        (indexes < 0)
        | (indexes >= len(x_train))
        | ~np.isfinite(distances)
    )
    bad_rows = np.flatnonzero(invalid.any(axis=1))
    if not len(bad_rows):
        return distances, indexes, 0

    repaired_distances = np.asarray(distances).copy()
    repaired_indexes = np.asarray(indexes, dtype=np.int64).copy()
    exact = NearestNeighbors(
        n_neighbors=neighbors, metric=metric, n_jobs=jobs,
    ).fit(x_train)
    for start in range(0, len(bad_rows), batch_size):
        selected = bad_rows[start : start + batch_size]
        exact_distances, exact_indexes = exact.kneighbors(
            x_query[selected], return_distance=True,
        )
        repaired_distances[selected] = exact_distances
        repaired_indexes[selected] = exact_indexes
    return repaired_distances, repaired_indexes, int(len(bad_rows))


def _select_knn(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    jobs: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    scaler = StandardScaler().fit(x_fit)
    fit = scaler.transform(x_fit).astype(np.float32)
    validation = scaler.transform(x_validation).astype(np.float32)
    rows: list[dict[str, object]] = []
    for metric in KNN_METRICS:
        maximum = min(max(K_VALUES), len(fit))
        distances, indexes, backend = _kneighbors(
            fit, validation, metric=metric, neighbors=maximum, jobs=jobs,
        )
        neighbor_labels = y_fit[indexes].astype(np.float64)
        for neighbors in K_VALUES:
            count = min(int(neighbors), len(fit))
            current_distances = distances[:, :count]
            current_labels = neighbor_labels[:, :count]
            for weights in KNN_WEIGHTS:
                if weights == "uniform":
                    probabilities = current_labels.mean(axis=1)
                else:
                    inverse = 1.0 / np.maximum(current_distances, 1e-12)
                    probabilities = np.sum(inverse * current_labels, axis=1) / np.sum(inverse, axis=1)
                    exact = current_distances <= 1e-12
                    has_exact = exact.any(axis=1)
                    if has_exact.any():
                        probabilities[has_exact] = (
                            np.sum(exact[has_exact] * current_labels[has_exact], axis=1)
                            / np.sum(exact[has_exact], axis=1)
                        )
                rows.append({
                    "metric": metric,
                    "neighbors": int(neighbors),
                    "weights": weights,
                    "backend": backend,
                    "validation_accuracy": _accuracy(y_validation, probabilities),
                })
    winner = max(
        rows,
        key=lambda row: (
            float(row["validation_accuracy"]),
            -int(row["neighbors"]),
            row["weights"] == "uniform",
            row["metric"] == "euclidean",
        ),
    )
    return winner, rows


def _select_logistic(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    scaler = StandardScaler().fit(x_fit)
    fit = scaler.transform(x_fit)
    validation = scaler.transform(x_validation)
    rows = []
    for regularization in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(
            C=regularization, max_iter=1000, solver="lbfgs", random_state=42,
        ).fit(fit, y_fit)
        accuracy = _accuracy(y_validation, model.predict_proba(validation)[:, 1])
        rows.append({"C": regularization, "validation_accuracy": accuracy})
    winner = max(rows, key=lambda row: (row["validation_accuracy"], -row["C"]))
    return float(winner["C"]), rows


def _final_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    regularization: float,
) -> np.ndarray:
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        C=regularization, max_iter=1000, solver="lbfgs", random_state=42,
    ).fit(scaler.transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_test))[:, 1]


def _cluster_features(
    signature: np.ndarray,
    cluster_model: MiniBatchKMeans,
    encoder: OneHotEncoder,
) -> np.ndarray:
    labels = cluster_model.predict(signature)
    return np.concatenate(
        (signature, cluster_model.transform(signature), encoder.transform(labels[:, None])),
        axis=1,
    ).astype(np.float32, copy=False)


def _group_response_clusters(
    metrics: dict[str, np.ndarray], group_positions: list[np.ndarray], group_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    responses = []
    for key in ("label_delta", "context_delta", "mask_delta"):
        values = np.asarray(metrics[key])
        responses.append(np.stack([values[selected].mean(axis=0) for selected in group_positions]))
    normalized = []
    for values in responses:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normalized.append(np.divide(values, norms, out=np.zeros_like(values), where=norms > 0))
    design = np.concatenate(normalized, axis=1)
    candidates = []
    trees = {}
    for count in range(2, len(group_names)):
        labels, tree = cluster_token_responses(responses, cluster_count=count)
        trees[count] = tree
        unique = np.unique(labels)
        score = (
            float(silhouette_score(design, labels, metric="cosine"))
            if 1 < len(unique) < len(labels) else -1.0
        )
        candidates.append({"k": count, "silhouette": score})
    best = max(candidates, key=lambda row: (row["silhouette"], -row["k"]))
    labels, tree = cluster_token_responses(responses, cluster_count=int(best["k"]))
    frame = pd.DataFrame({
        "semantic_group": group_names,
        "group_cluster": labels,
        "selected_k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
    })
    return frame, tree


def run(args: argparse.Namespace) -> dict[str, object]:
    metadata_model = getattr(args, "metadata_model", None) or args.model
    cache = args.cache_root / metadata_model / args.prompt_length
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    matches = [row for row in manifest["windows"] if int(row["test_year"]) == args.test_year]
    if len(matches) != 1:
        raise ValueError(f"test year {args.test_year} is unavailable")
    window = matches[0]
    if args.variant not in manifest["variants"]:
        raise ValueError(f"variant {args.variant} is absent from cache")
    frame = pd.read_parquet(cache / "rows.parquet")
    positions = _scope_positions(frame, window, args.target)
    if any(not len(value) for value in positions.values()):
        raise ValueError("one or more rolling scopes contain no finite target rows")

    span_cache_root = getattr(args, "span_cache_root", None)
    span_only = span_cache_root is not None
    token_rows = pd.DataFrame()
    group_rows = pd.DataFrame()
    token_tree = None
    if span_only:
        span_cache = span_cache_root / args.model / args.prompt_length
        span_manifest = json.loads((span_cache / "manifest.json").read_text(encoding="utf-8"))
        if args.variant not in span_manifest["variants"]:
            raise ValueError(f"variant {args.variant} is absent from span cache")
        if int(span_manifest["rows"]) != len(frame):
            raise ValueError("span cache and mechanism cache row counts differ")
    else:
        if args.baseline_root is None:
            raise ValueError("--baseline-root is required without --span-cache-root")
        baseline_path = args.baseline_root / args.model / f"{args.prompt_length}.npz"
        with np.load(baseline_path) as archive:
            prompt_only = np.asarray(archive["prompt_only"], dtype=np.float64)
        with np.load(cache / f"{args.variant}_token_moments.npz") as archive:
            moments = aggregate_moments(archive, window["fit_years"], target=args.target)
        with np.load(cache / "mask_delta_moments.npz") as archive:
            years = np.asarray(archive["years"], dtype=int)
            selected = [int(np.flatnonzero(years == int(year))[0]) for year in window["fit_years"]]
            delta_count = int(np.asarray(archive["delta_count"])[selected].sum())
            mask_delta = np.asarray(archive["delta_sum"])[selected].sum(axis=0) / delta_count
        token_metrics = token_metrics_from_moments(
            moments, prompt_only=prompt_only, mask_delta_mean=mask_delta,
        )
        fisher_order = np.lexsort((np.arange(len(token_metrics["fisher"])), -token_metrics["fisher"]))
        fisher_rank = np.empty(len(fisher_order), dtype=np.int32)
        fisher_rank[fisher_order] = np.arange(1, len(fisher_order) + 1)
        token_cluster_count = min(6, len(fisher_order) - 1)
        token_clusters, token_tree = cluster_token_responses(
            [token_metrics["label_delta"], token_metrics["context_delta"], token_metrics["mask_delta"]],
            cluster_count=token_cluster_count,
        )
        token_rows = pd.DataFrame({
            "model": args.model,
            "prompt_length": args.prompt_length,
            "variant": args.variant,
            "target": args.target,
            "test_year": args.test_year,
            "position_zero_based": np.arange(len(fisher_order)),
            "token": manifest["prompt_tokens"],
            "semantic_group": manifest["position_groups"],
            "fisher": token_metrics["fisher"],
            "fisher_rank": fisher_rank,
            "variance": token_metrics["variance_score"],
            "context_cosine_distance": token_metrics["context_cosine_distance"],
            "context_standardized_l2": token_metrics["context_standardized_l2"],
            "mask_delta_l2": token_metrics["mask_delta_l2"],
            "token_cluster": token_clusters,
        })
        group_rows = token_rows.groupby("semantic_group", sort=False).agg(
            token_count=("token", "size"),
            fisher_mean=("fisher", "mean"),
            fisher_max=("fisher", "max"),
            fisher_rank_best=("fisher_rank", "min"),
            context_cosine_mean=("context_cosine_distance", "mean"),
            context_standardized_l2_mean=("context_standardized_l2", "mean"),
            mask_delta_l2_mean=("mask_delta_l2", "mean"),
        ).reset_index()
        for name, value in (
            ("model", args.model), ("prompt_length", args.prompt_length),
            ("variant", args.variant), ("target", args.target),
            ("test_year", args.test_year),
        ):
            group_rows.insert(0, name, value)

    destination = (
        args.output_root / args.model / args.prompt_length / args.variant
        / args.target / str(args.test_year)
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite completed fold: {destination}")
    stage = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    stage.mkdir(parents=True)
    try:
        if not span_only:
            token_rows.to_parquet(stage / "token_metrics.parquet", index=False)
            group_rows.to_parquet(stage / "semantic_group_metrics.parquet", index=False)
            pd.DataFrame(
                token_tree, columns=["left", "right", "distance", "count"],
            ).to_csv(stage / "token_cluster_linkage.csv", index=False)
        report: dict[str, object] = {
            "format_version": "minimal_prompt_knn_fold_v1",
            "model": args.model,
            "prompt_length": args.prompt_length,
            "variant": args.variant,
            "target": args.target,
            "test_year": args.test_year,
            "window": window,
            "rows": {name: int(len(value)) for name, value in positions.items()},
            "classification": [],
            "representation_scope": "return_span" if span_only else "prompt_groups",
        }

        if args.prompt_length in ("short", "long"):
            if span_only:
                analysis_groups = ("return_span",)
                group_indexes = np.asarray([0], dtype=np.int64)
                groups = np.load(
                    span_cache / f"{args.variant}_return_span.npy", mmap_mode="r",
                )
                report.update({
                    "phrase": span_manifest["phrase"],
                    "tokens": span_manifest["tokens"],
                })
            else:
                analysis_groups = (
                    LONG_GROUPS
                    if args.prompt_length == "long"
                    else SHORT_GROUPS
                )
                missing = [name for name in analysis_groups if name not in manifest["groups"]]
                if missing:
                    raise ValueError(f"cache lacks semantic groups: {missing}")
                group_indexes = np.asarray(
                    [manifest["groups"].index(name) for name in analysis_groups], dtype=np.int64,
                )
                semantic_positions = [
                    np.asarray(manifest["group_positions"][name], dtype=np.int64)
                    for name in analysis_groups
                ]
                response_frame, response_tree = _group_response_clusters(
                    token_metrics, semantic_positions, list(analysis_groups),
                )
                response_frame.to_csv(stage / "semantic_group_clusters.csv", index=False)
                pd.DataFrame(
                    response_tree, columns=["left", "right", "distance", "count"],
                ).to_csv(stage / "semantic_group_cluster_linkage.csv", index=False)
                groups = np.load(
                    cache / f"{args.variant}_group_embeddings.npy", mmap_mode="r",
                )
            pca_fit = _fit_group_pca(
                groups, positions["fit"], group_indexes,
                components=args.group_components, sample_vectors=args.pca_sample_vectors,
                seed=42,
            )
            signature_fit = _group_signature(groups, positions["fit"], group_indexes, pca_fit)
            signature_validation = _group_signature(
                groups, positions["validation"], group_indexes, pca_fit,
            )
            signature_scaler = StandardScaler().fit(signature_fit)
            cluster_fit = signature_scaler.transform(signature_fit).astype(np.float32)
            cluster_validation = signature_scaler.transform(signature_validation).astype(np.float32)
            selected_k, k_rows = select_stable_kmeans(
                cluster_fit, cluster_validation,
                k_values=range(2, 13), seeds=SEEDS, sample_size=args.silhouette_sample,
            )
            pd.DataFrame(k_rows).to_csv(stage / "news_cluster_k_selection.csv", index=False)
            validation_cluster_model = MiniBatchKMeans(
                n_clusters=selected_k, random_state=42, batch_size=2048,
                n_init=3, max_iter=200,
            ).fit(cluster_fit)
            encoder = OneHotEncoder(
                categories=[np.arange(selected_k)], sparse_output=False,
                handle_unknown="ignore", dtype=np.float32,
            ).fit(np.arange(selected_k)[:, None])
            augmented_fit = _cluster_features(cluster_fit, validation_cluster_model, encoder)
            augmented_validation = _cluster_features(
                cluster_validation, validation_cluster_model, encoder,
            )
            y_fit = _labels(frame, args.target, positions["fit"])
            y_validation = _labels(frame, args.target, positions["validation"])

            selected_models = {}
            validation_rows = []
            for name, fit_values, validation_values in (
                ("group_pca_knn", cluster_fit, cluster_validation),
                ("cluster_augmented_knn", augmented_fit, augmented_validation),
            ):
                winner, rows = _select_knn(
                    fit_values, y_fit, validation_values, y_validation, jobs=args.jobs,
                )
                selected_models[name] = winner
                validation_rows.extend({"representation": name, **row} for row in rows)
            for name, fit_values, validation_values in (
                ("group_pca_logistic", cluster_fit, cluster_validation),
                ("cluster_augmented_logistic", augmented_fit, augmented_validation),
            ):
                regularization, rows = _select_logistic(
                    fit_values, y_fit, validation_values, y_validation,
                )
                selected_models[name] = {"C": regularization}
                validation_rows.extend({"representation": name, **row} for row in rows)
            pd.DataFrame(validation_rows).to_csv(stage / "classifier_validation.csv", index=False)

            pca_final = _fit_group_pca(
                groups, positions["all_train"], group_indexes,
                components=args.group_components, sample_vectors=args.pca_sample_vectors,
                seed=42,
            )
            signature_all = _group_signature(
                groups, positions["all_train"], group_indexes, pca_final,
            )
            signature_test = _group_signature(groups, positions["test"], group_indexes, pca_final)
            final_signature_scaler = StandardScaler().fit(signature_all)
            cluster_all = final_signature_scaler.transform(signature_all).astype(np.float32)
            cluster_test = final_signature_scaler.transform(signature_test).astype(np.float32)
            final_cluster_model = MiniBatchKMeans(
                n_clusters=selected_k, random_state=42, batch_size=2048,
                n_init=3, max_iter=200,
            ).fit(cluster_all)
            augmented_all = _cluster_features(cluster_all, final_cluster_model, encoder)
            augmented_test = _cluster_features(cluster_test, final_cluster_model, encoder)
            y_all = _labels(frame, args.target, positions["all_train"])
            y_test = _labels(frame, args.target, positions["test"])
            predictions = {}
            for name, train_values, test_values in (
                ("group_pca_knn", cluster_all, cluster_test),
                ("cluster_augmented_knn", augmented_all, augmented_test),
            ):
                selected_model = selected_models[name]
                scaler = StandardScaler().fit(train_values)
                predictions[name] = _neighbor_probabilities(
                    scaler.transform(train_values).astype(np.float32), y_all,
                    scaler.transform(test_values).astype(np.float32),
                    metric=str(selected_model["metric"]),
                    neighbors=int(selected_model["neighbors"]),
                    weights=str(selected_model["weights"]), jobs=args.jobs,
                )
            for name, train_values, test_values in (
                ("group_pca_logistic", cluster_all, cluster_test),
                ("cluster_augmented_logistic", augmented_all, augmented_test),
            ):
                predictions[name] = _final_logistic(
                    train_values, y_all, test_values, float(selected_models[name]["C"]),
                )
            all_clusters = final_cluster_model.predict(cluster_all)
            test_clusters = final_cluster_model.predict(cluster_test)
            global_rate = float(y_all.mean())
            cluster_rates = np.full(selected_k, global_rate, dtype=np.float64)
            for cluster in range(selected_k):
                selected_cluster = all_clusters == cluster
                if selected_cluster.any():
                    cluster_rates[cluster] = float(y_all[selected_cluster].mean())
            predictions["cluster_majority"] = cluster_rates[test_clusters]
            predictions["majority"] = np.full(len(y_test), global_rate)

            classification = []
            for name, probabilities in predictions.items():
                classification.append({
                    "representation": name,
                    "test_accuracy": _accuracy(y_test, probabilities),
                    "test_rows": int(len(y_test)),
                    "selected": selected_models.get(name),
                })
            prediction_frame = frame.iloc[positions["test"]][
                ["row_index", "entry_date", "stock_id", args.target]
            ].copy()
            prediction_frame["label"] = y_test
            prediction_frame["news_cluster"] = test_clusters
            for name, probabilities in predictions.items():
                prediction_frame[name] = probabilities
            prediction_frame.to_parquet(stage / "predictions.parquet", index=False)
            cluster_composition = prediction_frame.groupby("news_cluster").agg(
                rows=("label", "size"), positive_rate=("label", "mean"),
                stocks=("stock_id", "nunique"),
            ).reset_index()
            cluster_composition.to_csv(stage / "news_cluster_composition.csv", index=False)
            report.update({
                "selected_news_clusters": selected_k,
                "group_pca_components_each": int(pca_fit.n_components_),
                "semantic_groups": list(analysis_groups),
                "classification": classification,
            })
            if args.prompt_length == "long":
                report["long_groups"] = list(analysis_groups)

        atomic_json(stage / "metrics.json", report)
        (stage / "COMPLETED").write_text("minimal_prompt_knn_fold_v1\n", encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(destination)
        return {**report, "output": str(destination)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--span-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument(
        "--metadata-model", choices=("roberta", "bge_m3"),
        help="Model directory supplying model-independent rows and rolling windows",
    )
    parser.add_argument("--prompt-length", choices=("short", "long"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--test-year", type=int, choices=range(2018, 2027), required=True)
    parser.add_argument("--group-components", type=int, default=4)
    parser.add_argument("--pca-sample-vectors", type=int, default=20000)
    parser.add_argument("--silhouette-sample", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if min(args.group_components, args.pca_sample_vectors, args.silhouette_sample, args.jobs) < 1:
        raise ValueError("numeric controls must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
