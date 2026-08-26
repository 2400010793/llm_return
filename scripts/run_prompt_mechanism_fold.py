"""Run one leakage-safe prompt mechanism fold from a compact cache."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pooled_embedding_classification import (
    final_epoch_budget,
    fit_mlp_classifier_fixed_schedule,
    fit_mlp_classifier_with_validation,
)
from src.analysis.prompt_token_mechanisms import (
    SEEDS,
    aggregate_moments,
    cluster_token_responses,
    incremental_pca_transform,
    select_stable_kmeans,
    token_metrics_from_moments,
)
from src.evaluation.classification import (
    evaluate_binary_classification,
    paired_classification_comparison,
)


TARGETS = ("event_return_3d", "next_day_return")


def _indices(frame: pd.DataFrame, window: dict[str, object]) -> dict[str, np.ndarray]:
    years = pd.to_datetime(frame["entry_date"]).dt.year.to_numpy()
    return {
        "fit": np.flatnonzero(np.isin(years, window["fit_years"])),
        "validation": np.flatnonzero(np.isin(years, window["validation_years"])),
        "all_train": np.flatnonzero(np.isin(years, window["all_train_years"])),
        "test": np.flatnonzero(years == int(window["test_year"])),
    }


def _finite_labels(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric, np.isfinite(numeric)


def _fit_logistic(
    x_fit: np.ndarray, y_fit: pd.Series, x_validation: np.ndarray,
    x_all: np.ndarray, y_all: pd.Series, x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fit_values, fit_mask = _finite_labels(y_fit)
    all_values, all_mask = _finite_labels(y_all)
    fit_labels = (fit_values[fit_mask] > 0).astype(np.int8)
    all_labels = (all_values[all_mask] > 0).astype(np.int8)
    if len(np.unique(fit_labels)) < 2 or len(np.unique(all_labels)) < 2:
        raise ValueError("logistic fold requires both target classes")
    fit_scaler = StandardScaler().fit(x_fit[fit_mask])
    validation_model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=500, solver="lbfgs",
        random_state=42,
    ).fit(fit_scaler.transform(x_fit[fit_mask]), fit_labels)
    validation_probabilities = validation_model.predict_proba(
        fit_scaler.transform(x_validation),
    )[:, 1]
    all_scaler = StandardScaler().fit(x_all[all_mask])
    final_model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=500, solver="lbfgs",
        random_state=42,
    ).fit(all_scaler.transform(x_all[all_mask]), all_labels)
    test_probabilities = final_model.predict_proba(all_scaler.transform(x_test))[:, 1]
    return validation_probabilities, test_probabilities


def _cluster_projection(
    groups: np.ndarray, train_positions: np.ndarray, predict_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_parts = []
    predict_parts = []
    for group_index in range(groups.shape[1]):
        train, predict, _ = incremental_pca_transform(
            np.asarray(groups[train_positions, group_index, :], dtype=np.float32),
            np.asarray(groups[predict_positions, group_index, :], dtype=np.float32),
            components=4,
        )
        train_parts.append(train); predict_parts.append(predict)
    train_design = np.concatenate(train_parts, axis=1)
    predict_design = np.concatenate(predict_parts, axis=1)
    return incremental_pca_transform(
        train_design, predict_design, components=32,
    )[:2]


def _cluster_body_features(
    groups: np.ndarray, body: np.ndarray, positions: dict[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray], int, list[dict[str, float]], np.ndarray, np.ndarray,
]:
    fit_cluster, validation_cluster = _cluster_projection(
        groups, positions["fit"], positions["validation"],
    )
    selected_k, k_rows = select_stable_kmeans(
        fit_cluster, validation_cluster, seeds=SEEDS,
    )
    fit_model = MiniBatchKMeans(
        n_clusters=selected_k, random_state=42, batch_size=2048,
        n_init=3, max_iter=200,
    ).fit(fit_cluster)
    fit_labels = fit_model.predict(fit_cluster)
    validation_labels = fit_model.predict(validation_cluster)
    all_cluster, test_cluster = _cluster_projection(
        groups, positions["all_train"], positions["test"],
    )
    final_model = MiniBatchKMeans(
        n_clusters=selected_k, random_state=42, batch_size=2048,
        n_init=3, max_iter=200,
    ).fit(all_cluster)
    all_labels = final_model.predict(all_cluster)
    test_labels = final_model.predict(test_cluster)
    encoder = OneHotEncoder(
        categories=[np.arange(selected_k)], sparse_output=False,
        handle_unknown="ignore", dtype=np.float32,
    ).fit(np.arange(selected_k).reshape(-1, 1))
    labels_by_scope = {
        "fit": fit_labels, "validation": validation_labels,
        "all_train": all_labels, "test": test_labels,
    }
    output = {}
    for scope, selected in positions.items():
        one_hot = encoder.transform(labels_by_scope[scope].reshape(-1, 1))
        output[scope] = np.concatenate(
            (np.asarray(body[selected], dtype=np.float32), one_hot), axis=1,
        )
    return output, selected_k, k_rows, test_labels, test_cluster[:, :2]


def _representation_arrays(
    name: str, groups: np.ndarray, prompt_mean: np.ndarray, topk: np.ndarray,
    group_names: list[str], group_counts: np.ndarray,
) -> np.ndarray:
    if name == "full_prompt_mean":
        return prompt_mean
    if name == "top8":
        return topk
    kind, group = name.split("::", 1)
    index = group_names.index(group)
    if kind == "group":
        return groups[:, index, :]
    if kind == "leaveout":
        retained = int(group_counts.sum() - group_counts[index])
        if retained < 1:
            raise ValueError(f"cannot leave out the only prompt group: {group}")
        return (
            prompt_mean * float(group_counts.sum())
            - groups[:, index, :] * group_counts[index]
        ) / retained
    raise ValueError(f"unknown representation: {name}")


def _accuracy(values: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    return evaluate_binary_classification(values, probabilities, threshold=0.5)


def run(args: argparse.Namespace) -> dict[str, object]:
    cache = args.cache_root / args.model / args.prompt_length
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != args.model or manifest["prompt_length"] != args.prompt_length:
        raise ValueError("cache manifest identity mismatch")
    variant = args.variant
    if variant not in manifest["variants"]:
        raise ValueError(f"variant {variant} is absent from cache")
    windows = manifest["windows"]
    matches = [row for row in windows if int(row["test_year"]) == args.test_year]
    if len(matches) != 1:
        raise ValueError(f"test year {args.test_year} is unavailable or ambiguous")
    window = matches[0]
    window_index = windows.index(window)
    frame = pd.read_parquet(cache / "rows.parquet")
    positions = _indices(frame, window)
    groups = np.load(cache / f"{variant}_group_embeddings.npy", mmap_mode="r")
    prompt_mean = np.load(cache / f"{variant}_prompt_mean.npy", mmap_mode="r")
    body = np.load(cache / f"{variant}_body_mean.npy", mmap_mode="r")
    topk_all = np.load(
        cache / f"{variant}_top{manifest['topk']}_by_window.npy", mmap_mode="r",
    )
    target_index = TARGETS.index(args.target)
    topk = topk_all[target_index, window_index]
    group_names = list(manifest["groups"])
    group_counts = np.asarray([
        len(manifest["group_positions"][group]) for group in group_names
    ], dtype=np.float32)

    baseline_path = args.baseline_root / args.model / f"{args.prompt_length}.npz"
    with np.load(baseline_path) as baseline_archive:
        baseline = np.asarray(baseline_archive["prompt_only"], dtype=np.float64)
    with np.load(cache / f"{variant}_token_moments.npz") as archive:
        moments = aggregate_moments(archive, window["fit_years"], target=args.target)
    with np.load(cache / "mask_delta_moments.npz") as archive:
        stored_years = archive["years"].astype(int)
        selected_years = [
            int(np.flatnonzero(stored_years == int(year))[0])
            for year in window["fit_years"]
        ]
        count = int(archive["delta_count"][selected_years].sum())
        mask_delta = archive["delta_sum"][selected_years].sum(axis=0) / count
    token_metrics = token_metrics_from_moments(
        moments, prompt_only=baseline, mask_delta_mean=mask_delta,
    )
    token_order = np.lexsort((np.arange(len(token_metrics["fisher"])), -token_metrics["fisher"]))
    ranks = np.empty(len(token_order), dtype=np.int32); ranks[token_order] = np.arange(1, len(token_order) + 1)
    response_vectors = [token_metrics["label_delta"], token_metrics["context_delta"], mask_delta]
    token_clusters, token_tree = cluster_token_responses(
        response_vectors, cluster_count=min(len(group_names), len(token_order) - 1),
    )
    token_rows = pd.DataFrame({
        "model": args.model, "prompt_length": args.prompt_length, "variant": variant,
        "target": args.target, "test_year": args.test_year,
        "position_zero_based": np.arange(len(token_order)),
        "token": manifest["prompt_tokens"],
        "semantic_group": manifest["position_groups"],
        "fisher": token_metrics["fisher"], "fisher_rank": ranks,
        "variance": token_metrics["variance_score"],
        "context_cosine_distance": token_metrics["context_cosine_distance"],
        "context_standardized_l2": token_metrics["context_standardized_l2"],
        "mask_delta_l2": token_metrics["mask_delta_l2"],
        "token_cluster": token_clusters,
    })
    group_rows = token_rows.groupby("semantic_group", sort=False).agg(
        token_count=("token", "size"), fisher_mean=("fisher", "mean"),
        fisher_max=("fisher", "max"), fisher_rank_best=("fisher_rank", "min"),
        context_cosine_mean=("context_cosine_distance", "mean"),
        context_standardized_l2_mean=("context_standardized_l2", "mean"),
        mask_delta_l2_mean=("mask_delta_l2", "mean"),
    ).reset_index()

    cluster_features, selected_k, k_rows, test_clusters, test_coordinates = _cluster_body_features(
        groups, body, positions,
    )
    names = ["full_prompt_mean", "top8"]
    names.extend(f"group::{group}" for group in group_names)
    names.extend(f"leaveout::{group}" for group in group_names)
    classification_rows = []
    prediction_rows = []
    test_predictions: dict[str, np.ndarray] = {}
    best_validation = (-np.inf, "", None)
    target = args.target
    for name in names:
        matrix = _representation_arrays(
            name, groups, prompt_mean, topk, group_names, group_counts,
        )
        scoped = {scope: np.asarray(matrix[selected], dtype=np.float32) for scope, selected in positions.items()}
        validation_probabilities, test_probabilities = _fit_logistic(
            scoped["fit"], frame.iloc[positions["fit"]][target], scoped["validation"],
            scoped["all_train"], frame.iloc[positions["all_train"]][target], scoped["test"],
        )
        validation_metrics = _accuracy(
            frame.iloc[positions["validation"]][target], validation_probabilities,
        )
        test_metrics = _accuracy(frame.iloc[positions["test"]][target], test_probabilities)
        classification_rows.append({
            "representation": name, "classifier": "logistic",
            "validation_accuracy": validation_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_majority_accuracy": test_metrics["majority_accuracy"],
            "test_n": test_metrics["n"],
        })
        test_predictions[name] = test_probabilities
        if validation_metrics["accuracy"] > best_validation[0]:
            best_validation = (validation_metrics["accuracy"], name, scoped)
    validation_probabilities, test_probabilities = _fit_logistic(
        cluster_features["fit"], frame.iloc[positions["fit"]][target],
        cluster_features["validation"], cluster_features["all_train"],
        frame.iloc[positions["all_train"]][target], cluster_features["test"],
    )
    validation_metrics = _accuracy(
        frame.iloc[positions["validation"]][target], validation_probabilities,
    )
    test_metrics = _accuracy(frame.iloc[positions["test"]][target], test_probabilities)
    classification_rows.append({
        "representation": "cluster_plus_body", "classifier": "logistic",
        "validation_accuracy": validation_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_majority_accuracy": test_metrics["majority_accuracy"],
        "test_n": test_metrics["n"],
    })
    test_predictions["cluster_plus_body"] = test_probabilities
    if validation_metrics["accuracy"] > best_validation[0]:
        best_validation = (validation_metrics["accuracy"], "cluster_plus_body", cluster_features)

    baseline_predictions = test_predictions["full_prompt_mean"]
    test_frame = frame.iloc[positions["test"]].reset_index(drop=True)
    for row in classification_rows:
        name = str(row["representation"])
        probabilities = test_predictions[name]
        comparison = paired_classification_comparison(
            test_frame[target], baseline_predictions, probabilities,
            test_frame["entry_date"], n_bootstrap=args.bootstrap, seed=42,
            metrics=("accuracy",),
        )
        row["accuracy_delta_vs_full_prompt"] = comparison["delta"]["accuracy"]
        row["accuracy_delta_ci_low"] = comparison["clustered_95_ci"]["accuracy"][0]
        row["accuracy_delta_ci_high"] = comparison["clustered_95_ci"]["accuracy"][1]
        prediction_rows.append(pd.DataFrame({
            "row_index": test_frame["row_index"].to_numpy(),
            "entry_date": test_frame["entry_date"].to_numpy(),
            "stock_id": test_frame["stock_id"].astype(str).to_numpy(),
            "target": target, "actual_return": test_frame[target].to_numpy(),
            "representation": name, "classifier": "logistic",
            "probability": probabilities,
        }))

    mlp_summary = None
    if args.with_mlp:
        _, best_name, best_scoped = best_validation
        fit_values, fit_mask = _finite_labels(frame.iloc[positions["fit"]][target])
        fit_labels = (fit_values[fit_mask] > 0).astype(np.int8)
        candidate = {
            "hidden_layer_sizes": (64,), "alpha": 1e-3, "max_iter": 120,
            "batch_size": 256, "learning_rate_init": 1e-3,
            "n_iter_no_change": 10, "learning_rate_schedule": "constant",
        }
        _, validation_probabilities, _, selected_epoch, history = fit_mlp_classifier_with_validation(
            best_scoped["fit"][fit_mask], fit_labels, best_scoped["validation"],
            frame.iloc[positions["validation"]][target], candidate=candidate,
            seed=42, selection_metric="accuracy",
        )
        all_values, all_mask = _finite_labels(frame.iloc[positions["all_train"]][target])
        final_model = fit_mlp_classifier_fixed_schedule(
            best_scoped["all_train"][all_mask], (all_values[all_mask] > 0).astype(np.int8),
            candidate=candidate, selected_epoch=final_epoch_budget(selected_epoch), seed=42,
        )
        test_probabilities = final_model.predict_proba(best_scoped["test"])[:, 1]
        validation_metrics = _accuracy(
            frame.iloc[positions["validation"]][target], validation_probabilities,
        )
        test_metrics = _accuracy(test_frame[target], test_probabilities)
        classification_rows.append({
            "representation": best_name, "classifier": "simple_mlp",
            "validation_accuracy": validation_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_majority_accuracy": test_metrics["majority_accuracy"],
            "test_n": test_metrics["n"], "selected_epoch": selected_epoch,
        })
        prediction_rows.append(pd.DataFrame({
            "row_index": test_frame["row_index"].to_numpy(),
            "entry_date": test_frame["entry_date"].to_numpy(),
            "stock_id": test_frame["stock_id"].astype(str).to_numpy(),
            "target": target, "actual_return": test_frame[target].to_numpy(),
            "representation": best_name, "classifier": "simple_mlp",
            "probability": test_probabilities,
        }))
        mlp_summary = {
            "selected_representation": best_name, "selected_epoch": selected_epoch,
            "epochs_run": len(history), "history": history,
        }

    cluster_frame = test_frame[
        ["row_index", "entry_date", "stock_id", target]
        + (["industry"] if "industry" in test_frame else [])
    ].copy()
    cluster_frame["cluster"] = test_clusters
    cluster_frame["pca_1"] = test_coordinates[:, 0]
    cluster_frame["pca_2"] = (
        test_coordinates[:, 1] if test_coordinates.shape[1] > 1 else 0.0
    )
    if "industry" not in cluster_frame:
        cluster_frame["industry"] = "unknown"
    cluster_rows = []
    for cluster, values in cluster_frame.groupby("cluster", sort=True):
        returns = pd.to_numeric(values[target], errors="coerce")
        industries = values["industry"].fillna("unknown").astype(str).value_counts().head(5)
        cluster_rows.append({
            "cluster": int(cluster), "n": len(values),
            "finite_target_n": int(returns.notna().sum()),
            "up_rate": float((returns.dropna() > 0).mean()) if returns.notna().any() else np.nan,
            "unique_stocks": int(values["stock_id"].nunique()),
            "top_industries": json.dumps(industries.to_dict(), ensure_ascii=False),
        })

    output = (
        args.output_root / args.model / args.prompt_length / variant / args.target
        / f"test_{args.test_year}"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fold output: {output}")
    stage = output.with_name(f".{output.name}.partial.{os.getpid()}")
    stage.mkdir(parents=True)
    try:
        token_rows.to_parquet(stage / "token_metrics.parquet", index=False)
        group_rows.to_parquet(stage / "group_metrics.parquet", index=False)
        pd.DataFrame(classification_rows).to_parquet(
            stage / "classification_metrics.parquet", index=False,
        )
        pd.concat(prediction_rows, ignore_index=True).to_parquet(
            stage / "predictions.parquet", index=False,
        )
        pd.DataFrame(k_rows).to_parquet(stage / "k_selection.parquet", index=False)
        pd.DataFrame(cluster_rows).to_parquet(stage / "cluster_composition.parquet", index=False)
        cluster_frame.to_parquet(stage / "cluster_assignments.parquet", index=False)
        np.save(stage / "token_linkage.npy", token_tree)
        report = {
            "format_version": "prompt_mechanism_fold_v1", "model": args.model,
            "prompt_length": args.prompt_length, "variant": variant,
            "target": args.target, "window": window,
            "selected_cluster_k": selected_k, "cluster_seeds": list(SEEDS),
            "logistic_representations": names + ["cluster_plus_body"],
            "mlp": mlp_summary, "bootstrap": args.bootstrap,
            "claim_levels": {
                "descriptive": "token/group contextual shifts and clusters",
                "predictive": "rolling out-of-sample Accuracy and leave-one-group-out deltas",
                "causal": "not established until position-matched encoder controls complete",
            },
        }
        (stage / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        (stage / "COMPLETED").write_text("prompt_mechanism_fold_v1\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output)
        return {**report, "output": str(output)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-length", choices=("short", "long"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--with-mlp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
