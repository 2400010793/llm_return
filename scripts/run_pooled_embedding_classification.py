"""Classify one-day returns from one aligned pooled-embedding experiment cell.

The runner uses the paper-style 6-year fit, 2-year chronological validation,
and 1-year out-of-sample test design. PCA and feature scaling are fitted only
inside each training window. The primary protocol uses the same one-day
``next_day_return`` target for fitting and evaluation; the legacy three-day
event target remains available only through an explicit command-line override.
Predictions are persisted for paired tests.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import math
import os
import random
import resource
import sys
import time
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import FEATURE_GROUPS, align_embeddings_to_panel, discover_pooled_parts, load_pooled_embeddings
from src.evaluation.artifacts import (
    acquire_bundle_lock, atomic_joblib, atomic_json, build_input_fingerprints,
    completed_bundle_matches, experiment_id, file_fingerprint,
    release_bundle_lock, write_completed,
)
from src.evaluation.classification import evaluate_binary_classification
from src.models.dimension_reduction import fit_reduce
from src.models.dynamic_token_classifier import DynamicTokenGateTransformer
from src.models.representation_models import (
    TemporalEarlyStoppingMLPClassifier,
    _classifier,
)
from src.models.token_gating import fit_token_gate


SUPPORTED_CLASSIFIERS = (
    "logistic", "linear_svm", "sgd", "mlp", "simple_mlp", "lstm",
    "random_forest", "extra_trees", "hist_gradient_boosting",
    "xgboost", "lightgbm", "catboost", "knn",
)
SUPPORTED_REDUCERS = (
    "none", "pca", "token_gate_pca", "group_gate", "group_gate_pca",
    "dynamic_token_gate",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def learning_rate_at_epoch(
    base_learning_rate: float,
    epoch: int,
    max_iter: int,
    schedule: str = "constant",
    *,
    min_ratio: float = 0.1,
) -> float:
    """Return a deterministic learning rate for one epoch.

    ``cosine`` and ``inverse_time`` deliberately keep a non-zero floor. This
    slows convergence without silently freezing the network before validation
    early stopping has had a chance to select an epoch.
    """
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if max_iter < 1 or not 1 <= epoch <= max_iter:
        raise ValueError("epoch must lie in [1, max_iter]")
    if not 0.0 < min_ratio <= 1.0:
        raise ValueError("min_ratio must lie in (0, 1]")
    if schedule == "constant":
        factor = 1.0
    elif schedule == "cosine":
        progress = (epoch - 1) / max(1, max_iter - 1)
        factor = min_ratio + (1.0 - min_ratio) * 0.5 * (
            1.0 + np.cos(np.pi * progress)
        )
    elif schedule == "inverse_time":
        progress = (epoch - 1) / max(1, max_iter - 1)
        factor = min_ratio + (1.0 - min_ratio) / (1.0 + 4.0 * progress)
    else:
        raise ValueError(f"unsupported learning-rate schedule: {schedule}")
    return float(base_learning_rate * factor)


def final_epoch_budget(selected_epoch: int, multiplier: float = 1.0) -> int:
    """Convert a validation-selected epoch into the final-fit budget."""
    if selected_epoch < 1 or multiplier <= 0:
        raise ValueError("selected_epoch must be positive and multiplier must be > 0")
    return max(1, int(math.ceil(selected_epoch * multiplier)))


def set_network_learning_rate(network, learning_rate: float) -> None:
    """Update sklearn's Adam optimizer before the next ``partial_fit`` call."""
    network.learning_rate_init = float(learning_rate)
    optimizer = getattr(network, "_optimizer", None)
    if optimizer is not None:
        optimizer.learning_rate_init = float(learning_rate)


def fit_mlp_classifier_fixed_schedule(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    candidate: dict[str, Any],
    selected_epoch: int,
    seed: int,
) -> TemporalEarlyStoppingMLPClassifier:
    """Refit on fit+validation data with the selected schedule and budget."""
    scaler = StandardScaler(with_mean=False).fit(x)
    transformed = scaler.transform(x)
    wrapper = TemporalEarlyStoppingMLPClassifier(
        hidden_layer_sizes=tuple(candidate["hidden_layer_sizes"]),
        alpha=float(candidate["alpha"]), max_iter=int(candidate["max_iter"]),
        batch_size=candidate.get("batch_size", 256),
        learning_rate_init=float(candidate.get("learning_rate_init", 1e-3)),
        random_state=seed,
    )
    network = wrapper._new_network(int(candidate["max_iter"]), one_epoch=True)
    classes = np.array([0, 1], dtype=np.int8)
    schedule = str(candidate.get("learning_rate_schedule", "constant"))
    schedule_horizon = max(int(candidate["max_iter"]), int(selected_epoch))
    for epoch in range(1, int(selected_epoch) + 1):
        set_network_learning_rate(
            network,
            learning_rate_at_epoch(
                wrapper.learning_rate_init, epoch, schedule_horizon, schedule,
            ),
        )
        network.partial_fit(transformed, labels, classes=classes)
    return TemporalEarlyStoppingMLPClassifier.from_fitted(
        network, scaler,
        hidden_layer_sizes=wrapper.hidden_layer_sizes,
        alpha=wrapper.alpha, max_iter=wrapper.max_iter,
        batch_size=wrapper.batch_size,
        learning_rate_init=wrapper.learning_rate_init,
        random_state=seed, selected_epoch=int(selected_epoch),
    )


def dense_grid(classifier: str, stage: str) -> dict[str, list[Any]]:
    fine = stage == "fine"
    if classifier == "simple_mlp" and stage == "stable":
        return {
            "hidden_layer_sizes": [(32,)],
            "alpha": [1e-3],
            "learning_rate_init": [3e-4],
            "batch_size": [256],
            "max_iter": [240],
            "solver": ["adam"],
            "early_stopping": [True],
            "validation_fraction": [0.1],
            "n_iter_no_change": [20],
        }
    if classifier == "logistic":
        return {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0] if fine else [0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
            "class_weight": [None, "balanced"],
        }
    if classifier == "linear_svm":
        return {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0] if fine else [0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        }
    if classifier == "sgd":
        return {
            "loss": ["log_loss"],
            "alpha": [1e-6, 1e-5, 1e-4],
            "class_weight": ["balanced"],
        }
    if classifier == "mlp":
        return {
            "hidden_layer_sizes": [(64,), (128, 32)] if not fine else [(32,), (64,), (128, 64)],
            "alpha": [1e-3] if not fine else [3e-4, 1e-3],
            "learning_rate_init": [1e-3] if not fine else [3e-4, 1e-3],
            "batch_size": [256] if not fine else [128, 256, 512],
            "max_iter": [240] if not fine else [300],
            "solver": ["adam"],
            "early_stopping": [False],
        }
    if classifier == "simple_mlp":
        # One deliberately small architecture: this is the inexpensive
        # nonlinear check, not another broad neural-network sweep.
        return {
            "hidden_layer_sizes": [(64,)] if not fine else [(32,), (64,)],
            "alpha": [1e-3] if not fine else [3e-4, 1e-3],
            "learning_rate_init": [1e-3] if not fine else [3e-4, 1e-3],
            "batch_size": [256] if not fine else [128, 256, 512],
            "max_iter": [120] if not fine else [240],
            "solver": ["adam"],
            "early_stopping": [True],
            "validation_fraction": [0.1],
            "n_iter_no_change": [10] if not fine else [8, 12],
        }
    if classifier == "lstm":
        # Keep the first LSTM comparison deliberately bounded.  The current
        # pooled matrix is row-level, so LSTMClassifier uses the paper-style
        # one-step shape [row, 1, features]; cross-day sequences need a
        # separate stock-day sequence builder rather than a silent protocol
        # change in this runner.
        return {
            "hidden_size": [64] if not fine else [32, 64, 128],
            "num_layers": [1],
            "dropout": [0.0] if not fine else [0.0, 0.1],
            "learning_rate": [1e-3] if not fine else [3e-4, 1e-3],
            "weight_decay": [1e-4],
            "epochs": [5] if not fine else [5, 10],
            "batch_size": [512],
            "gradient_clip_norm": [1.0],
            "device": ["auto"],
        }
    if classifier == "random_forest":
        return {
            "n_estimators": [300],
            "max_depth": [8, 16] if not fine else [6, 8, 12, 16, 24],
            "min_samples_leaf": [2, 10],
            "max_features": ["sqrt"],
        }
    if classifier == "extra_trees":
        return {
            "n_estimators": [500],
            "max_depth": [16, None] if not fine else [12, 20, None],
            "min_samples_leaf": [2, 10] if not fine else [1, 2, 5, 10],
            "max_features": ["sqrt"] if not fine else ["sqrt", 0.5],
            "class_weight": ["balanced"],
        }
    if classifier == "hist_gradient_boosting":
        # Histogram GBDT is the scalable tree model for the 350k-row dense
        # panel. The much slower full random-forest grid remains opt-in only.
        return {
            "learning_rate": [0.03, 0.08] if fine else [0.05],
            "max_iter": [300] if fine else [150],
            "max_leaf_nodes": [15, 31, 63] if fine else [15, 31],
            "min_samples_leaf": [50, 200] if fine else [100],
            "l2_regularization": [0.1, 1.0, 10.0] if fine else [0.1, 1.0],
            "early_stopping": [True],
            "class_weight": ["balanced"],
        }
    if classifier == "xgboost":
        return {
            "n_estimators": [600] if fine else [300],
            "learning_rate": [0.03, 0.08] if fine else [0.05],
            "max_depth": [3, 5, 7] if fine else [3, 6],
            "min_child_weight": [1.0, 10.0] if fine else [5.0],
            "subsample": [0.8], "colsample_bytree": [0.6, 0.9] if fine else [0.8],
            "reg_lambda": [1.0, 10.0] if fine else [1.0],
            "tree_method": ["hist"], "eval_metric": ["logloss"],
        }
    if classifier == "lightgbm":
        return {
            "n_estimators": [800] if fine else [400],
            "learning_rate": [0.02, 0.05] if fine else [0.05],
            "num_leaves": [15, 31, 63] if fine else [15, 31],
            "min_child_samples": [50, 200] if fine else [100],
            "subsample": [0.8], "colsample_bytree": [0.6, 0.9] if fine else [0.8],
            "reg_lambda": [1.0, 10.0] if fine else [1.0],
        }
    if classifier == "catboost":
        return {
            "iterations": [800] if fine else [400],
            "learning_rate": [0.02, 0.05] if fine else [0.05],
            "depth": [4, 6, 8] if fine else [4, 6],
            "l2_leaf_reg": [1.0, 5.0, 20.0] if fine else [3.0],
            "random_strength": [0.0, 1.0] if fine else [1.0],
            "loss_function": ["Logloss"], "verbose": [False],
        }
    if classifier == "knn":
        # Exact KNN is expensive for 350k rows. Keep the PCA experiment to a
        # single predeclared specification rather than repeating six massive
        # validation distance searches.
        return {
            "n_neighbors": [15],
            "weights": ["distance"],
            "metric": ["cosine"],
        }
    raise ValueError(f"unsupported dense classifier: {classifier}")


def finite_target(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(numeric)
    return numeric, mask


def evaluate(values: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    numeric, mask = finite_target(values)
    mask &= np.isfinite(probabilities)
    return evaluate_binary_classification(numeric[mask], probabilities[mask])


def validation_selection_score(
    metrics: dict[str, float], metric: str,
) -> float:
    """Return a maximization score for validation model selection."""
    if metric not in {"accuracy", "balanced_accuracy", "auc"}:
        raise ValueError(f"unsupported validation selection metric: {metric}")
    score = float(metrics[metric])
    return score if np.isfinite(score) else -np.inf


def scale_windows(x_fit, x_val, x_all, x_test, enabled: bool):
    if not enabled:
        return x_fit, x_val, x_all, x_test
    fit_scaler = StandardScaler()
    all_scaler = StandardScaler()
    return (
        fit_scaler.fit_transform(x_fit).astype(np.float32, copy=False),
        fit_scaler.transform(x_val).astype(np.float32, copy=False),
        all_scaler.fit_transform(x_all).astype(np.float32, copy=False),
        all_scaler.transform(x_test).astype(np.float32, copy=False),
    )


def token_gate_audit(gate) -> dict[str, Any]:
    return {
        "method": gate.method,
        "token_count": gate.token_count,
        "hidden_size": gate.hidden_size,
        "selected_positions_zero_based": gate.selected_positions.tolist(),
        "selected_positions_one_based": (gate.selected_positions + 1).tolist(),
        "selected_scores": gate.scores[gate.selected_positions].tolist(),
        "all_position_scores": gate.scores.tolist(),
        "output_dimension": int(len(gate.selected_positions) * gate.hidden_size),
        "aggregation": "none; selected full token vectors flattened in original position order",
    }


def preprocess_windows(
    x_fit, x_val, x_all, x_test, args, *,
    y_fit=None, y_all=None, prompt_token_shape: tuple[int, ...] | None = None,
    group_shape: tuple[int, int] | None = None,
):
    fit_reducer = all_reducer = None
    fit_gate = all_gate = None
    fit_dynamic_gate = all_dynamic_gate = None
    gate_audits = None
    gate_reducers = {"token_gate_pca", "group_gate", "group_gate_pca"}
    if args.reducer in gate_reducers:
        if args.reducer == "token_gate_pca":
            active_shape = prompt_token_shape
            gate_method = args.token_gate_method
            gate_keep = args.token_gate_keep
        else:
            active_shape = group_shape
            gate_method = args.group_gate_method
            gate_keep = args.group_gate_keep
        if active_shape is None or len(active_shape) != 2:
            raise ValueError(f"{args.reducer} requires a [groups, hidden] shape")
        if y_fit is None or y_all is None:
            raise ValueError(f"{args.reducer} requires fit and all-train targets")
        token_count = int(active_shape[0])
        fit_gate = fit_token_gate(
            x_fit, np.asarray(y_fit, dtype=float), token_count=token_count,
            keep_tokens=gate_keep, method=gate_method,
        )
        all_gate = fit_token_gate(
            x_all, np.asarray(y_all, dtype=float), token_count=token_count,
            keep_tokens=gate_keep, method=gate_method,
        )
        x_fit, x_val = fit_gate.transform(x_fit), fit_gate.transform(x_val)
        x_all, x_test = all_gate.transform(x_all), all_gate.transform(x_test)
        gate_audits = {"fit": token_gate_audit(fit_gate), "all_train": token_gate_audit(all_gate)}
    if args.reducer == "dynamic_token_gate":
        if prompt_token_shape is None or len(prompt_token_shape) != 2:
            raise ValueError("dynamic_token_gate requires a [tokens, hidden] shape")
        if y_fit is None or y_all is None:
            raise ValueError("dynamic_token_gate requires fit and all-train targets")
        token_count, hidden_size = map(int, prompt_token_shape)
        dynamic_kwargs = {
            "gate_hidden_size": getattr(args, "dynamic_gate_hidden_size", 64),
            "representation_size": getattr(args, "dynamic_gate_representation_size", 64),
            "gate_mode": getattr(args, "dynamic_gate_mode", "dynamic"),
            "dropout": getattr(args, "dynamic_gate_dropout", 0.1),
            "learning_rate": getattr(args, "dynamic_gate_learning_rate", 1e-3),
            "weight_decay": getattr(args, "dynamic_gate_weight_decay", 1e-4),
            "epochs": getattr(args, "dynamic_gate_epochs", 10),
            "batch_size": getattr(args, "dynamic_gate_batch_size", 256),
            "gradient_clip_norm": getattr(args, "dynamic_gate_gradient_clip_norm", 1.0),
            "random_state": args.seed,
            "device": getattr(args, "dynamic_gate_device", "auto"),
        }
        fit_input, all_input = x_fit, x_all
        fit_dynamic_gate = DynamicTokenGateTransformer(
            token_count, hidden_size, **dynamic_kwargs,
        ).fit(fit_input, np.asarray(y_fit, dtype=float))
        all_dynamic_gate = DynamicTokenGateTransformer(
            token_count, hidden_size, **dynamic_kwargs,
        ).fit(all_input, np.asarray(y_all, dtype=float))
        x_fit, _fit_weights = fit_dynamic_gate.transform(fit_input, return_weights=True)
        x_val = fit_dynamic_gate.transform(x_val)
        x_all, _all_weights = all_dynamic_gate.transform(all_input, return_weights=True)
        x_test = all_dynamic_gate.transform(x_test)
        gate_audits = {
            "fit": fit_dynamic_gate.audit(fit_input, scope="fit"),
            "all_train": all_dynamic_gate.audit(all_input, scope="all_train"),
        }
    if args.reducer in ("pca", "token_gate_pca", "group_gate_pca"):
        fit_result = fit_reduce(
            x_fit, x_val, method="pca", n_components=args.reducer_components,
            random_state=args.seed,
        )
        all_result = fit_reduce(
            x_all, x_test, method="pca", n_components=args.reducer_components,
            random_state=args.seed,
        )
        x_fit, x_val = fit_result.train, fit_result.predict
        x_all, x_test = all_result.train, all_result.predict
        fit_reducer, all_reducer = fit_result.reducer, all_result.reducer
    scale_free = {
        "mlp", "simple_mlp", "random_forest", "extra_trees",
        "hist_gradient_boosting", "xgboost", "lightgbm", "catboost",
    }
    scaler_enabled = args.scaler == "standard" or (
        args.scaler == "auto" and args.classifier not in scale_free
    )
    fit_scaler = all_scaler = None
    if scaler_enabled:
        fit_scaler = StandardScaler()
        all_scaler = StandardScaler()
        x_fit = fit_scaler.fit_transform(x_fit).astype(np.float32, copy=False)
        x_val = fit_scaler.transform(x_val).astype(np.float32, copy=False)
        x_all = all_scaler.fit_transform(x_all).astype(np.float32, copy=False)
        x_test = all_scaler.transform(x_test).astype(np.float32, copy=False)
    preprocessors = {
        "fit": {
            "token_gate": fit_gate,
            "dynamic_token_gate": fit_dynamic_gate,
            "reducer": fit_reducer,
            "scaler": fit_scaler,
        },
        "all_train": {
            "token_gate": all_gate,
            "dynamic_token_gate": all_dynamic_gate,
            "reducer": all_reducer,
            "scaler": all_scaler,
        },
    }
    return (x_fit, x_val, x_all, x_test, fit_reducer, all_reducer,
            scaler_enabled, preprocessors, gate_audits)


def fit_mlp_classifier_with_validation(
    x_fit: np.ndarray,
    fit_labels: np.ndarray,
    x_val: np.ndarray,
    y_val: pd.Series,
    *,
    candidate: dict[str, Any],
    seed: int,
    selection_metric: str = "balanced_accuracy",
    selection_min_epoch: int = 1,
) -> tuple[TemporalEarlyStoppingMLPClassifier, np.ndarray, float, int, list[dict[str, Any]]]:
    """Train one MLP and select its epoch on the chronological validation fold."""
    max_iter = int(candidate.get("max_iter", 120))
    patience = int(candidate.get("n_iter_no_change", 10))
    if not 1 <= selection_min_epoch <= max_iter:
        raise ValueError("selection_min_epoch must lie in [1, max_iter]")
    learning_rate_schedule = str(candidate.get("learning_rate_schedule", "constant"))
    wrapper = TemporalEarlyStoppingMLPClassifier(
        hidden_layer_sizes=tuple(candidate.get("hidden_layer_sizes", (64,))),
        alpha=float(candidate.get("alpha", 1e-3)), max_iter=max_iter,
        batch_size=candidate.get("batch_size", 256),
        learning_rate_init=float(candidate.get("learning_rate_init", 1e-3)),
        random_state=seed,
    )
    scaler = StandardScaler(with_mean=False).fit(x_fit)
    transformed_fit = scaler.transform(x_fit)
    transformed_val = scaler.transform(x_val)
    network = wrapper._new_network(max_iter, one_epoch=True)
    classes = np.array([0, 1], dtype=np.int8)
    best_network = None
    best_score = -np.inf
    best_epoch = 0
    best_probabilities = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_iter + 1):
        learning_rate = learning_rate_at_epoch(
            float(candidate.get("learning_rate_init", 1e-3)),
            epoch, max_iter, learning_rate_schedule,
        )
        set_network_learning_rate(network, learning_rate)
        network.partial_fit(transformed_fit, fit_labels, classes=classes)
        probabilities = network.predict_proba(transformed_val)[:, 1]
        validation_metrics = evaluate(y_val, probabilities)
        score = validation_selection_score(validation_metrics, selection_metric)
        eligible = epoch >= selection_min_epoch
        history.append({
            "epoch": int(epoch),
            "learning_rate": learning_rate,
            "train_loss": float(network.loss_),
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_balanced_accuracy": float(
                validation_metrics["balanced_accuracy"]
            ),
            "validation_log_loss": float(validation_metrics["log_loss"]),
            "validation_brier": float(validation_metrics["brier"]),
            "validation_auc": float(validation_metrics["auc"]),
            "selection_metric": selection_metric,
            "selection_score": score,
            "selection_eligible": eligible,
        })
        if eligible and score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            best_network = copy.deepcopy(network)
            best_probabilities = probabilities.copy()
            stale = 0
        elif eligible:
            stale += 1
        if stale >= patience:
            break
    if best_network is None or best_probabilities is None:
        raise RuntimeError("classification MLP early stopping produced no model")
    model = TemporalEarlyStoppingMLPClassifier.from_fitted(
        best_network, scaler,
        hidden_layer_sizes=wrapper.hidden_layer_sizes,
        alpha=wrapper.alpha, max_iter=max_iter,
        batch_size=wrapper.batch_size,
        learning_rate_init=wrapper.learning_rate_init,
        random_state=seed, selected_epoch=best_epoch,
    )
    return model, best_probabilities, best_score, best_epoch, history


def fit_one(x_fit, y_fit, x_val, y_val, x_all, y_all, x_test, args):
    fit_values, fit_mask = finite_target(y_fit)
    all_values, all_mask = finite_target(y_all)
    fit_labels = (fit_values[fit_mask] > 0).astype(np.int8)
    all_labels = (all_values[all_mask] > 0).astype(np.int8)
    if len(np.unique(fit_labels)) < 2 or len(np.unique(all_labels)) < 2:
        raise ValueError("training windows must contain both classes")

    best_score = -np.inf
    best_params: dict[str, Any] = {}
    best_history: list[dict[str, Any]] | None = None
    search_history: list[dict[str, Any]] = []
    temporal_mlp = args.classifier in {"mlp", "simple_mlp"}
    selection_metric = str(
        getattr(args, "early_stopping_metric", "balanced_accuracy")
    )
    selection_min_epoch = int(
        getattr(args, "early_stopping_min_epoch", 1)
    )
    for candidate in ParameterGrid(dense_grid(args.classifier, args.search_stage)):
        if temporal_mlp:
            if getattr(args, "max_iter_override", None) is not None:
                candidate["max_iter"] = int(args.max_iter_override)
            if getattr(args, "patience_override", None) is not None:
                candidate["n_iter_no_change"] = int(args.patience_override)
            model, probabilities, score, selected_epoch, history = (
                fit_mlp_classifier_with_validation(
                    x_fit[fit_mask], fit_labels, x_val, y_val,
                    candidate=dict(candidate), seed=args.seed,
                    selection_metric=selection_metric,
                    selection_min_epoch=selection_min_epoch,
                )
            )
            candidate = dict(candidate)
            candidate["learning_rate_schedule"] = getattr(
                args, "learning_rate_schedule", "constant"
            )
            candidate["selected_epoch"] = selected_epoch
            candidate["early_stopping"] = "chronological_validation"
            candidate["early_stopping_min_epoch"] = selection_min_epoch
            candidate["early_stopping_patience"] = int(
                candidate.get("n_iter_no_change", 10)
            )
        else:
            model = _classifier(args.classifier, args.seed, **candidate)
            model.fit(x_fit[fit_mask], fit_labels)
            probabilities = model.predict_proba(x_val)[:, 1]
            score = validation_selection_score(
                evaluate(y_val, probabilities), selection_metric
            )
            history = None
        if score > best_score:
            best_score = score
            best_params = dict(candidate)
            best_history = history
        search_history.append({
            "params": dict(candidate),
            "selected_epoch": int(selected_epoch) if temporal_mlp else None,
            "selection_metric": selection_metric,
            "best_validation_score": float(score),
            "epochs_run": len(history) if history is not None else None,
            "stopped_early": bool(
                temporal_mlp and history is not None
                and len(history) < int(candidate.get("max_iter", 0))
            ),
            "epoch_history": history,
        })

    if temporal_mlp:
        selected_epoch = int(best_params["selected_epoch"])
        validation_model = fit_mlp_classifier_with_validation(
            x_fit[fit_mask], fit_labels, x_val, y_val,
            candidate=best_params, seed=args.seed,
            selection_metric=selection_metric,
            selection_min_epoch=selection_min_epoch,
        )[0]
    else:
        validation_model = _classifier(args.classifier, args.seed, **best_params)
        validation_model.fit(x_fit[fit_mask], fit_labels)
    validation_probabilities = validation_model.predict_proba(x_val)[:, 1]
    if temporal_mlp:
        final_epoch = final_epoch_budget(
            selected_epoch,
            float(getattr(args, "final_epoch_multiplier", 1.0)),
        )
        final_model = fit_mlp_classifier_fixed_schedule(
            x_all[all_mask], all_labels,
            candidate=best_params,
            selected_epoch=final_epoch,
            seed=args.seed,
        )
        best_params["final_fit_epoch"] = final_epoch
        best_params["final_epoch_multiplier"] = float(
            getattr(args, "final_epoch_multiplier", 1.0)
        )
    else:
        final_model = _classifier(args.classifier, args.seed, **best_params)
        final_model.fit(x_all[all_mask], all_labels)
    return (
        final_model.predict_proba(x_test)[:, 1], best_params, best_score,
        validation_probabilities, validation_model, final_model, best_history,
        search_history,
    )


def experiment_spec(args, parts: list[Path]) -> dict[str, Any]:
    spec = {
        "format_version": "pooled_classification_bundle_v1",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        },
        "inputs": build_input_fingerprints(
            args.panel,
            parts,
            include_prompt_tokens=args.feature.startswith("prompt_tokens"),
        ),
        "design": {
            "fit_years": 6, "validation_years": 2, "test_years": 1,
            "run_mode": args.run_mode,
            "requested_test_years": args.test_years,
            "train_target": args.train_target_column,
            "evaluation_target": args.evaluation_target_column,
            "date_column": args.date_column,
            "panel_row_index_column": args.panel_row_index_column,
            "filter_column": args.filter_column or None,
        },
        "experiment": {
            "model": args.model, "variant": args.variant, "feature": args.feature,
            "classifier": args.classifier, "reducer": args.reducer,
            "reducer_components": args.reducer_components,
            "token_gate_method": args.token_gate_method,
            "token_gate_keep": args.token_gate_keep,
            "group_gate_method": args.group_gate_method,
            "group_gate_keep": args.group_gate_keep,
            "scaler": args.scaler, "search_stage": args.search_stage,
            "learning_rate_schedule": args.learning_rate_schedule,
            "early_stopping_metric": args.early_stopping_metric,
            "early_stopping_min_epoch": args.early_stopping_min_epoch,
            "final_epoch_multiplier": args.final_epoch_multiplier,
            "seed": args.seed,
        },
    }
    if args.classifier == "lstm":
        spec["experiment"]["lstm"] = {
            "sequence_length": 1,
            "protocol": "paper-style one-step LSTM over each frozen embedding row",
            "cross_day_context": False,
        }
    if args.classifier in {"mlp", "simple_mlp"}:
        spec["experiment"]["mlp_search_grid"] = dense_grid(
            args.classifier, args.search_stage
        )
        spec["experiment"]["mlp_training_logging"] = {
            "per_epoch": [
                "learning_rate", "train_loss", "validation_accuracy",
                "validation_balanced_accuracy", "validation_log_loss",
                "validation_brier", "validation_auc", "selection_score",
                "selection_eligible",
            ],
            "early_stopping_metric": args.early_stopping_metric,
            "early_stopping_min_epoch": args.early_stopping_min_epoch,
            "patience": "candidate.n_iter_no_change",
            "final_fit": "train_plus_validation_for_selected_epoch_budget",
        }
    if args.reducer == "dynamic_token_gate":
        spec["experiment"]["dynamic_gate"] = {
            "mode": args.dynamic_gate_mode,
            "hidden_size": args.dynamic_gate_hidden_size,
            "representation_size": args.dynamic_gate_representation_size,
            "dropout": args.dynamic_gate_dropout,
            "learning_rate": args.dynamic_gate_learning_rate,
            "weight_decay": args.dynamic_gate_weight_decay,
            "epochs": args.dynamic_gate_epochs,
            "batch_size": args.dynamic_gate_batch_size,
            "gradient_clip_norm": args.dynamic_gate_gradient_clip_norm,
            "device": args.dynamic_gate_device,
            "training_protocol": "fit-only gate and separately refit all-train gate",
        }
    return spec


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=(
            "roberta", "bge_m3", "ckip_bert", "xlm_roberta_large",
            "qwen3_embedding_8b", "finbert2_base",
        ),
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=("short", "masked_short", "long", "masked_long", "plain"),
        required=True,
    )
    parser.add_argument(
        "--feature",
        choices=tuple(FEATURE_GROUPS),
        required=True,
    )
    parser.add_argument("--classifier", choices=SUPPORTED_CLASSIFIERS, required=True)
    parser.add_argument(
        "--reducer",
        choices=SUPPORTED_REDUCERS,
        default="none",
    )
    parser.add_argument("--reducer-components", type=int, default=128)
    parser.add_argument(
        "--token-gate-method",
        choices=("fisher", "variance", "logistic_l1"), default="fisher",
    )
    parser.add_argument("--token-gate-keep", type=int, default=8)
    parser.add_argument(
        "--group-gate-method",
        choices=("fisher", "variance", "logistic_l1"), default="fisher",
    )
    parser.add_argument("--group-gate-keep", type=int, default=2)
    parser.add_argument(
        "--dynamic-gate-mode", choices=("dynamic", "static", "uniform"),
        default="dynamic",
    )
    parser.add_argument("--dynamic-gate-hidden-size", type=int, default=64)
    parser.add_argument("--dynamic-gate-representation-size", type=int, default=64)
    parser.add_argument("--dynamic-gate-dropout", type=float, default=0.1)
    parser.add_argument("--dynamic-gate-learning-rate", type=float, default=1e-3)
    parser.add_argument("--dynamic-gate-weight-decay", type=float, default=1e-4)
    parser.add_argument("--dynamic-gate-epochs", type=int, default=10)
    parser.add_argument("--dynamic-gate-batch-size", type=int, default=256)
    parser.add_argument("--dynamic-gate-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--dynamic-gate-device", default="auto")
    parser.add_argument("--scaler", choices=("auto", "none", "standard"), default="auto")
    parser.add_argument(
        "--train-target-column", default="next_day_return",
        help="One-day return target used to fit the classifier; event_return_3d is legacy-only.",
    )
    parser.add_argument(
        "--evaluation-target-column", default="next_day_return",
        help="One-day return target used for validation and out-of-sample evaluation.",
    )
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--panel-row-index-column", default="row_index")
    parser.add_argument("--filter-column", default="", help="Optional boolean eligibility column")
    parser.add_argument("--expected-embedding-rows", type=int, default=350577)
    parser.add_argument(
        "--max-embedding-matrix-gib", type=float, default=64.0,
        help="Fail before loading when summary shapes imply a larger dense matrix",
    )
    parser.add_argument(
        "--search-stage", choices=("coarse", "fine", "stable"), default="coarse"
    )
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "cosine", "inverse_time"),
        default="constant",
        help="MLP schedule applied per epoch; non-MLP classifiers ignore it.",
    )
    parser.add_argument(
        "--early-stopping-metric",
        choices=("accuracy", "balanced_accuracy", "auc"),
        default="balanced_accuracy",
        help=(
            "Chronological-validation metric used to select MLP epochs and "
            "hyperparameters; final reports still include raw Accuracy."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-epoch", type=int, default=1,
        help=(
            "Burn-in epoch count: epochs before this value are logged but "
            "cannot be selected and do not consume early-stopping patience."
        ),
    )
    parser.add_argument(
        "--final-epoch-multiplier", type=float, default=1.0,
        help="Multiply validation-selected epoch only for final fit+validation training.",
    )
    parser.add_argument(
        "--run-mode", choices=("screen", "final-test"), default="final-test",
        help="screen evaluates only 2018-2023 fit and 2024-2025 validation data.",
    )
    parser.add_argument(
        "--test-years", nargs="+", type=int, default=None,
        help="Optional final-test years to run; training windows remain unchanged.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    if args.final_epoch_multiplier <= 0:
        raise ValueError("--final-epoch-multiplier must be positive")
    if args.early_stopping_min_epoch < 1:
        raise ValueError("--early-stopping-min-epoch must be positive")
    if args.search_stage == "stable" and args.classifier != "simple_mlp":
        raise ValueError("--search-stage stable is only defined for simple_mlp")
    if args.run_mode == "screen" and args.test_years:
        raise ValueError("--test-years is only valid in final-test mode")
    if args.reducer in ("pca", "token_gate_pca", "group_gate_pca") and args.reducer_components < 1:
        raise ValueError("reducer-components must be positive")
    if args.reducer == "token_gate_pca" and args.feature != "prompt_tokens_flat":
        raise ValueError("token_gate_pca is only valid with prompt_tokens_flat")
    if args.reducer == "dynamic_token_gate" and args.feature != "prompt_tokens_flat":
        raise ValueError("dynamic_token_gate is only valid with prompt_tokens_flat")
    if args.reducer in ("group_gate", "group_gate_pca") and args.feature not in {
        "title_body_concat", "title_body_full_concat",
    }:
        raise ValueError("group gates require a segment-concat feature")
    if args.reducer == "dynamic_token_gate":
        if args.dynamic_gate_hidden_size < 1 or args.dynamic_gate_representation_size < 1:
            raise ValueError("dynamic gate sizes must be positive")
        if not 0.0 <= args.dynamic_gate_dropout < 1.0:
            raise ValueError("dynamic-gate-dropout must lie in [0, 1)")
        if args.dynamic_gate_learning_rate <= 0 or args.dynamic_gate_weight_decay < 0:
            raise ValueError("dynamic gate learning rate/weight decay are invalid")
        if args.dynamic_gate_epochs < 1 or args.dynamic_gate_batch_size < 1:
            raise ValueError("dynamic gate epochs and batch size must be positive")
        if args.dynamic_gate_gradient_clip_norm <= 0:
            raise ValueError("dynamic-gate-gradient-clip-norm must be positive")
    seed_everything(args.seed)

    parts = discover_pooled_parts(args.embedding_root, args.model, args.variant)
    if not parts:
        raise ValueError(
            f"no complete embedding parts under {args.embedding_root} for "
            f"{args.model}/{args.variant}"
        )
    spec = experiment_spec(args, parts)
    spec_id = experiment_id(spec)
    bundle = args.artifact_dir or args.output.with_suffix(".artifacts")
    if not args.force_recompute and completed_bundle_matches(bundle, spec_id):
        if not args.output.is_file():
            raise ValueError(f"completed artifact bundle exists but report is missing: {args.output}")
        print(json.dumps({
            "output": str(args.output), "artifact_dir": str(bundle),
            "experiment_id": spec_id, "resumed": True,
        }, ensure_ascii=False))
        return
    writer_lock = acquire_bundle_lock(bundle)
    atexit.register(release_bundle_lock, writer_lock)
    atomic_json(bundle / "spec.json", {**spec, "experiment_id": spec_id})

    panel = pd.read_parquet(args.panel)
    required = {
        args.panel_row_index_column, args.date_column,
        args.train_target_column, args.evaluation_target_column,
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {', '.join(sorted(missing))}")
    embeddings = load_pooled_embeddings(
        args.embedding_root,
        model=args.model,
        variant=args.variant,
        feature=args.feature,
        require_complete_rows=args.expected_embedding_rows or None,
        max_matrix_gib=args.max_embedding_matrix_gib,
    )
    frame, matrix = align_embeddings_to_panel(
        panel, embeddings, panel_row_index_column=args.panel_row_index_column
    )
    if len(frame) != len(panel):
        raise ValueError(f"only {len(frame)}/{len(panel)} panel rows have embeddings")
    alignment_audit = {
        "method": "exact_row_index",
        "panel_key": args.panel_row_index_column,
        "panel_rows": len(panel),
        "embedding_rows": len(embeddings.metadata),
        "matched_rows": len(frame),
        "unmatched_panel_rows": len(panel) - len(frame),
        "embedding_row_index_min": int(embeddings.metadata["row_index"].min()),
        "embedding_row_index_max": int(embeddings.metadata["row_index"].max()),
        "embedding_row_index_unique": bool(embeddings.metadata["row_index"].is_unique),
    }
    if args.filter_column:
        if args.filter_column not in frame:
            raise ValueError(f"filter column missing: {args.filter_column}")
        keep = frame[args.filter_column].fillna(False).astype(bool).to_numpy()
        frame, matrix = frame.loc[keep].copy(), matrix[keep]

    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
    valid_date = frame[args.date_column].notna().to_numpy()
    frame, matrix = frame.loc[valid_date].copy(), matrix[valid_date]
    order = np.argsort(frame[args.date_column].to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    matrix = matrix[order]
    frame["year"] = frame[args.date_column].dt.year
    years = sorted(int(year) for year in frame["year"].dropna().unique())
    required_years = 8 if args.run_mode == "screen" else 9
    if len(years) < required_years:
        raise ValueError(
            f"{args.run_mode} design needs at least {required_years} calendar years; "
            f"found {years}"
        )

    results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    positions: list[int | None] | range = (
        [None] if args.run_mode == "screen" else range(8, len(years))
    )
    if args.run_mode == "final-test" and args.test_years:
        requested = set(args.test_years)
        available = {years[position] for position in positions}
        missing_years = sorted(requested.difference(available))
        if missing_years:
            raise ValueError(f"requested test years are unavailable: {missing_years}")
        positions = [position for position in positions if years[position] in requested]
    for test_position in positions:
        if test_position is None:
            test_year = None
            in_years = years[:8]
        else:
            test_year = years[test_position]
            in_years = years[test_position - 8:test_position]
        fit_years, validation_years = in_years[:6], in_years[6:]
        fit_idx = np.flatnonzero(frame["year"].isin(fit_years).to_numpy())
        val_idx = np.flatnonzero(frame["year"].isin(validation_years).to_numpy())
        if args.run_mode == "screen":
            # Reuse fit/validation inputs in the helper's all/test slots. The
            # screen branch persists only fit-window preprocessing and
            # validation predictions; it never selects or evaluates 2026.
            all_idx, test_idx = fit_idx, val_idx
        else:
            all_idx = np.flatnonzero(frame["year"].isin(in_years).to_numpy())
            test_idx = np.flatnonzero(frame["year"].eq(test_year).to_numpy())
        if not all((len(fit_idx), len(val_idx), len(all_idx), len(test_idx))):
            continue

        preprocess_started = time.perf_counter()
        component_shapes = embeddings.component_shapes
        group_shape = None
        if len(component_shapes) > 1 and len({shape for shape in component_shapes}) == 1:
            first_shape = component_shapes[0]
            if len(first_shape) == 1:
                group_shape = (len(component_shapes), int(first_shape[0]))
        x_fit, x_val, x_all, x_test, fit_reducer, all_reducer, scaled, preprocessors, gate_audits = preprocess_windows(
            matrix[fit_idx], matrix[val_idx], matrix[all_idx], matrix[test_idx], args,
            y_fit=frame.iloc[fit_idx][args.train_target_column],
            y_all=frame.iloc[all_idx][args.train_target_column],
            prompt_token_shape=(embeddings.component_shapes[0]
                                if len(embeddings.component_shapes) == 1 else None),
            group_shape=group_shape,
        )
        preprocess_seconds = time.perf_counter() - preprocess_started
        fit_started = time.perf_counter()
        probabilities, params, validation_accuracy, validation_probabilities, validation_model, final_model, early_stopping_history, hyperparameter_search_history = fit_one(
            x_fit, frame.iloc[fit_idx][args.train_target_column],
            x_val, frame.iloc[val_idx][args.evaluation_target_column],
            x_all, frame.iloc[all_idx][args.train_target_column],
            x_test, args,
        )
        fit_seconds = time.perf_counter() - fit_started
        fold_bundle = bundle / (
            "screen" if args.run_mode == "screen" else f"test_year_{test_year}"
        )
        atomic_joblib(fold_bundle / "fit_preprocessor.joblib", preprocessors["fit"])
        atomic_joblib(fold_bundle / "validation_model.joblib", validation_model)
        if args.classifier in {"mlp", "simple_mlp"}:
            atomic_json(
                fold_bundle / "mlp_training_history.json",
                {
                    "classifier": args.classifier,
                    "selected_params": params,
                    "selected_epoch_history": early_stopping_history,
                    "hyperparameter_search": hyperparameter_search_history,
                    "final_train_loss_curve": [
                        float(value) for value in getattr(final_model, "loss_curve_", [])
                    ],
                },
            )
        validation_metrics = evaluate(
            frame.iloc[val_idx][args.evaluation_target_column], validation_probabilities
        )
        validation_predictions = frame.iloc[val_idx][
            [args.panel_row_index_column, args.date_column, args.evaluation_target_column]
        ].copy()
        validation_predictions["test_year"] = test_year
        validation_predictions["probability"] = validation_probabilities
        validation_predictions.to_parquet(
            fold_bundle / "validation_predictions.parquet", index=False
        )
        if args.run_mode == "screen":
            results.append({
                "test_year": None,
                "fit_years": fit_years,
                "validation_years": validation_years,
                "model": args.model,
                "variant": args.variant,
                "prompt_condition": embeddings.prompt_condition,
                "feature": args.feature,
                "representation": (
                    f"{args.model}:{args.variant}:"
                    f"{embeddings.prompt_condition}:{args.feature}"
                ),
                "classifier": args.classifier,
                "run_mode": args.run_mode,
                "reducer": args.reducer,
                "reducer_components": (
                    args.reducer_components
                    if args.reducer in ("pca", "token_gate_pca", "group_gate_pca")
                    else None
                ),
                "token_gate": gate_audits,
                "dynamic_token_gate": gate_audits if args.reducer == "dynamic_token_gate" else None,
                "scaled": scaled,
                "best_params": params,
                "early_stopping_history": early_stopping_history,
                "hyperparameter_search_history": hyperparameter_search_history,
                "final_train_loss_curve": [
                    float(value) for value in getattr(final_model, "loss_curve_", [])
                ],
                "validation_selection_metric": args.early_stopping_metric,
                "validation_selection_score": validation_accuracy,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_metrics": validation_metrics,
                "n_fit": len(fit_idx),
                "n_validation": len(val_idx),
                "preprocess_seconds": preprocess_seconds,
                "model_selection_seconds": fit_seconds,
                "fit_explained_variance": (
                    float(fit_reducer.explained_variance_ratio_.sum())
                    if fit_reducer is not None else None
                ),
            })
            continue

        atomic_joblib(fold_bundle / "all_train_preprocessor.joblib", preprocessors["all_train"])
        atomic_joblib(fold_bundle / "final_model.joblib", final_model)
        restored_model = joblib.load(fold_bundle / "final_model.joblib")
        restored_probabilities = restored_model.predict_proba(x_test)[:, 1]
        if not np.allclose(restored_probabilities, probabilities, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"reloaded final model changed probabilities for test year {test_year}")
        metrics = evaluate(frame.iloc[test_idx][args.evaluation_target_column], probabilities)
        results.append({
            "test_year": test_year,
            "fit_years": fit_years,
            "validation_years": validation_years,
            "model": args.model,
            "variant": args.variant,
            "prompt_condition": embeddings.prompt_condition,
            "feature": args.feature,
            "representation": f"{args.model}:{args.variant}:{embeddings.prompt_condition}:{args.feature}",
            "classifier": args.classifier,
            "reducer": args.reducer,
            "reducer_components": args.reducer_components if args.reducer in ("pca", "token_gate_pca", "group_gate_pca") else None,
            "token_gate": gate_audits,
            "dynamic_token_gate": gate_audits if args.reducer == "dynamic_token_gate" else None,
            "scaled": scaled,
            "best_params": params,
            "early_stopping_history": early_stopping_history,
            "hyperparameter_search_history": hyperparameter_search_history,
            "final_train_loss_curve": [
                float(value) for value in getattr(final_model, "loss_curve_", [])
            ],
            "validation_selection_metric": args.early_stopping_metric,
            "validation_selection_score": validation_accuracy,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_metrics": validation_metrics,
            "n_fit": len(fit_idx),
            "n_validation": len(val_idx),
            "n_all_train": len(all_idx),
            "n_test_rows": len(test_idx),
            "preprocess_seconds": preprocess_seconds,
            "model_selection_and_final_fit_seconds": fit_seconds,
            "fit_explained_variance": float(fit_reducer.explained_variance_ratio_.sum()) if fit_reducer is not None else None,
            "all_train_explained_variance": float(all_reducer.explained_variance_ratio_.sum()) if all_reducer is not None else None,
            **metrics,
        })
        predictions = frame.iloc[test_idx][
            [args.panel_row_index_column, args.date_column, args.evaluation_target_column]
        ].copy()
        predictions["test_year"] = test_year
        predictions["probability"] = probabilities
        actual = pd.to_numeric(
            predictions[args.evaluation_target_column], errors="coerce"
        ).to_numpy(dtype=float)
        predictions["actual_label"] = np.where(np.isfinite(actual), (actual > 0).astype(np.int8), np.nan)
        prediction_frames.append(predictions)
    if not results:
        raise ValueError(f"no {args.run_mode} folds were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_suffix(".predictions.parquet")
    if args.run_mode == "final-test":
        pd.concat(prediction_frames, ignore_index=True).to_parquet(
            prediction_path, index=False
        )
    report = {
        "design": {
            "fit_years": 6,
            "validation_years": 2,
            "test_years": 1,
            "run_mode": args.run_mode,
            "requested_test_years": args.test_years,
            "train_target": args.train_target_column,
            "evaluation_target": args.evaluation_target_column,
            "date_column": args.date_column,
            "seed": args.seed,
        },
        "input": {
            "panel": str(args.panel),
            "embedding_root": str(args.embedding_root),
            "embedding_parts": [str(path) for path in embeddings.parts],
            "embedding_rows": len(embeddings.metadata),
            "classification_rows": len(frame),
            "embedding_dimension": int(matrix.shape[1]),
            "embedding_matrix_gib": float(matrix.nbytes / 1024 ** 3),
            "embedding_alignment": alignment_audit,
        },
        "experiment": {
            "model": args.model,
            "variant": args.variant,
            "feature": args.feature,
            "prompt_condition": embeddings.prompt_condition,
            "classifier": args.classifier,
            "reducer": args.reducer,
            "reducer_components": args.reducer_components if args.reducer in ("pca", "token_gate_pca", "group_gate_pca") else None,
            "token_gate_method": args.token_gate_method if args.reducer == "token_gate_pca" else None,
            "token_gate_keep": args.token_gate_keep if args.reducer == "token_gate_pca" else None,
            "group_gate_method": args.group_gate_method if args.reducer.startswith("group_gate") else None,
            "group_gate_keep": args.group_gate_keep if args.reducer.startswith("group_gate") else None,
            "dynamic_gate": (
                {
                    "mode": args.dynamic_gate_mode,
                    "hidden_size": args.dynamic_gate_hidden_size,
                    "representation_size": args.dynamic_gate_representation_size,
                    "epochs": args.dynamic_gate_epochs,
                    "training_protocol": "fit-only gate and separately refit all-train gate",
                }
                if args.reducer == "dynamic_token_gate" else None
            ),
            "scaler": args.scaler,
            "filter_column": args.filter_column or None,
            "lstm": (
                {
                    "sequence_length": 1,
                    "protocol": "paper-style one-step LSTM over each frozen embedding row",
                    "cross_day_context": False,
                }
                if args.classifier == "lstm" else None
            ),
        },
        "years": years,
        "predictions": (
            str(prediction_path) if args.run_mode == "final-test" else None
        ),
        "artifact_bundle": str(bundle),
        "experiment_id": spec_id,
        "results": results,
        "runtime": {
            "total_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "provenance": {
            "task_record_id": os.environ.get("TASK_RECORD_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    atomic_json(args.output, report)
    atomic_json(bundle / "report.json", report)
    artifact_files = sorted(
        path for path in bundle.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "COMPLETED", ".writer.lock"}
    )
    atomic_json(bundle / "manifest.json", {
        "experiment_id": spec_id,
        "report": str(args.output),
        "predictions": (
            str(prediction_path) if args.run_mode == "final-test" else None
        ),
        "test_years": [
            int(row["test_year"]) for row in results
            if row["test_year"] is not None
        ],
        "task_record_id": os.environ.get("TASK_RECORD_ID"),
        "files": {
            str(path.relative_to(bundle)): file_fingerprint(path, hash_content=True)
            for path in artifact_files
        },
        "external_outputs": {
            "report": file_fingerprint(args.output, hash_content=True),
            **({
                "predictions": file_fingerprint(prediction_path, hash_content=True),
            } if args.run_mode == "final-test" else {}),
        },
    })
    write_completed(bundle, spec_id)
    release_bundle_lock(writer_lock)
    atexit.unregister(release_bundle_lock)
    print(json.dumps({
        "output": str(args.output), "folds": len(results),
        "predictions": (
            str(prediction_path) if args.run_mode == "final-test" else None
        ), "artifact_dir": str(bundle),
        "experiment_id": spec_id, "resumed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
