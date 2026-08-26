"""Reusable components for leakage-safe continuous-return prediction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.evaluation.prediction_metrics import daily_rank_ic, return_prediction_metrics
from src.models.dimension_reduction import fit_reduce


@dataclass(frozen=True)
class AlphaSelection:
    """Validation result and fitted model for a selected Ridge penalty."""

    alpha: float
    model: Ridge
    predictions: np.ndarray
    metrics: dict[str, float]
    audit: list[dict[str, Any]]


@dataclass(frozen=True)
class RegressorSelection:
    """Validation-selected continuous-return regressor and its audit trail."""

    regressor: str
    params: dict[str, Any]
    model: Any
    predictions: np.ndarray
    metrics: dict[str, float]
    audit: list[dict[str, Any]]


class TargetScaledMLPRegressor:
    """MLP regressor with explicit target scaling and a fixed epoch budget.

    The wrapper deliberately does not use sklearn's random internal
    ``early_stopping`` split.  The rolling runner selects ``selected_epoch``
    from the chronological validation window, then refits this estimator on
    fit+validation data for exactly that many epochs before the test fold.
    """

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...],
        alpha: float,
        max_iter: int,
        seed: int = 42,
        batch_size: int | str = "auto",
        learning_rate_init: float = 1e-3,
    ) -> None:
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.batch_size = batch_size
        self.learning_rate_init = float(learning_rate_init)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TargetScaledMLPRegressor":
        values = np.asarray(y, dtype=float)
        if values.ndim != 1 or len(values) != len(X) or not np.isfinite(values).all():
            raise ValueError("y must be a finite vector aligned with X")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.target_scaler_ = StandardScaler().fit(values.reshape(-1, 1))
        scaled = self.target_scaler_.transform(values.reshape(-1, 1)).ravel()
        self.network_ = _new_fixed_epoch_mlp(self, self.max_iter)
        self.network_.fit(X, scaled)
        self.n_iter_ = int(self.network_.n_iter_)
        self.loss_curve_ = list(self.network_.loss_curve_)
        self.selected_epoch_ = self.n_iter_
        return self

    @classmethod
    def from_fitted(
        cls,
        network: MLPRegressor,
        target_scaler: StandardScaler,
        *,
        hidden_layer_sizes: tuple[int, ...],
        alpha: float,
        max_iter: int,
        seed: int,
        batch_size: int | str,
        learning_rate_init: float,
        selected_epoch: int,
    ) -> "TargetScaledMLPRegressor":
        model = cls(
            hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
            max_iter=max_iter, seed=seed, batch_size=batch_size,
            learning_rate_init=learning_rate_init,
        )
        model.network_ = network
        model.target_scaler_ = target_scaler
        model.n_iter_ = int(selected_epoch)
        model.selected_epoch_ = int(selected_epoch)
        model.loss_curve_ = list(getattr(network, "loss_curve_", []))
        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "network_"):
            raise RuntimeError("model must be fitted before predict")
        scaled = np.asarray(self.network_.predict(X), dtype=float).reshape(-1, 1)
        return self.target_scaler_.inverse_transform(scaled).ravel()


def _new_fixed_epoch_mlp(
    spec: Any, max_iter: int, *, one_epoch: bool = False,
) -> MLPRegressor:
    """Create an MLP whose stopping is controlled by the caller."""
    def value(name: str, default: Any = None) -> Any:
        if isinstance(spec, Mapping):
            return spec.get(name, default)
        return getattr(spec, name, default)

    seed = int(value("seed", 42))
    return MLPRegressor(
        hidden_layer_sizes=tuple(value("hidden_layer_sizes")),
        alpha=float(value("alpha")),
        max_iter=1 if one_epoch else int(max_iter),
        batch_size=value("batch_size", "auto"),
        solver="adam",
        learning_rate_init=float(value("learning_rate_init", 1e-3)),
        early_stopping=False,
        # Disable sklearn's loss-based early stop.  The rolling validation
        # window, or the selected validation epoch, controls termination.
        tol=0.0,
        n_iter_no_change=max(2, int(max_iter) + 1),
        random_state=seed,
        warm_start=bool(one_epoch),
    )


def fit_mlp_with_validation(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    validation_frame: pd.DataFrame,
    *,
    params: Mapping[str, Any],
    stock_column: str,
    date_column: str,
    target_column: str,
    min_stocks_per_day: int,
    selection_correlation: str = "spearman",
) -> tuple[TargetScaledMLPRegressor, np.ndarray, int, list[dict[str, float]]]:
    """Fit one MLP with chronological validation-based early stopping."""
    values = np.asarray(y_fit, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("MLP fit targets must be finite")
    max_iter = int(params["max_iter"])
    patience = int(params.get("early_stopping_patience", 10))
    if max_iter < 1 or patience < 1:
        raise ValueError("MLP max_iter and early-stopping patience must be positive")

    target_scaler = StandardScaler().fit(values.reshape(-1, 1))
    scaled = target_scaler.transform(values.reshape(-1, 1)).ravel()
    network = _new_fixed_epoch_mlp(params, max_iter, one_epoch=True)
    best_network: MLPRegressor | None = None
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    best_predictions: np.ndarray | None = None
    stale = 0
    history: list[dict[str, float]] = []
    historical_mean = float(np.mean(values))

    for epoch in range(1, max_iter + 1):
        network.partial_fit(x_fit, scaled)
        scaled_predictions = network.predict(x_validation).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(scaled_predictions).ravel()
        metrics, _ = evaluate_stock_day_predictions(
            validation_frame, predictions, historical_mean=historical_mean,
            stock_column=stock_column, date_column=date_column,
            target_column=target_column, min_stocks_per_day=min_stocks_per_day,
            correlation_method=selection_correlation,
        )
        history.append({
            "epoch": float(epoch),
            "rank_ic_mean": float(metrics.get("rank_ic_mean", np.nan)),
            "oos_r2_vs_historical_mean": float(
                metrics.get("oos_r2_vs_historical_mean", np.nan)
            ),
            "mse": float(metrics.get("mse", np.nan)),
        })
        score_key = _selection_key(metrics)
        if best_key is None or score_key > best_key:
            best_key = score_key
            best_network = copy.deepcopy(network)
            best_epoch = epoch
            best_predictions = predictions.copy()
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_network is None or best_predictions is None:
        raise RuntimeError("MLP validation early stopping produced no model")
    model = TargetScaledMLPRegressor.from_fitted(
        best_network, target_scaler,
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        alpha=float(params["alpha"]), max_iter=max_iter,
        seed=int(params.get("seed", 42)),
        batch_size=params.get("batch_size", "auto"),
        learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
        selected_epoch=best_epoch,
    )
    return model, best_predictions, best_epoch, history


def finite_target(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Convert a target series to floats and return its finite-value mask."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric, np.isfinite(numeric)


def aggregate_stock_day_predictions(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    stock_column: str,
    date_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Average announcement predictions and reject inconsistent stock-day labels."""
    if len(frame) != len(predictions):
        raise ValueError("frame and predictions must have equal lengths")
    required = {stock_column, date_column, target_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing stock-day columns: {', '.join(sorted(missing))}")
    values = frame[[stock_column, date_column, target_column]].copy()
    values["prediction"] = np.asarray(predictions, dtype=float)
    values[target_column] = pd.to_numeric(values[target_column], errors="coerce")
    values = values.dropna(subset=[stock_column, date_column, target_column, "prediction"])
    groups = values.groupby([stock_column, date_column], sort=True, observed=True)
    allow_inconsistent = target_column in {
        "event_abs_return_3d", "abnormal_event_return_3d", "event_return_3d",
        "volume_shock", "range_shock", "post_realized_volatility_5d",
        "post_realized_volatility_20d", "volatility_jump", "volatility_jump_ratio_v2",
    }
    inconsistent = groups[target_column].nunique(dropna=False).gt(1)
    if inconsistent.any() and not allow_inconsistent:
        examples = inconsistent[inconsistent].index.tolist()[:10]
        raise ValueError(f"inconsistent targets within stock-day groups: {examples}")
    target_agg = "mean" if allow_inconsistent else "first"
    return groups.agg(
        actual_return=(target_column, target_agg),
        prediction=("prediction", "mean"),
        n_announcements=("prediction", "size"),
    ).reset_index()


def fit_preprocessor(
    x_train: np.ndarray,
    x_predict: np.ndarray,
    *,
    reducer: str,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit optional reduction and scaling on one training window only."""
    reduction = None
    if reducer == "pca":
        result = fit_reduce(
            x_train, x_predict, method="pca", n_components=components,
            random_state=seed,
        )
        x_train, x_predict, reduction = result.train, result.predict, result.reducer
    elif reducer != "none":
        raise ValueError("reducer must be one of: none, pca")
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    transformed_predict = scaler.transform(x_predict).astype(np.float32, copy=False)
    return transformed_train, transformed_predict, {"reducer": reduction, "scaler": scaler}


def evaluate_stock_day_predictions(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    historical_mean: float,
    stock_column: str,
    date_column: str,
    target_column: str,
    min_stocks_per_day: int,
    correlation_method: str = "spearman",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Aggregate announcement predictions and compute return/rank metrics."""
    stock_day = aggregate_stock_day_predictions(
        frame, predictions, stock_column=stock_column,
        date_column=date_column, target_column=target_column,
    )
    metrics = return_prediction_metrics(
        stock_day["actual_return"].to_numpy(), stock_day["prediction"].to_numpy(),
        historical_mean=historical_mean,
    )
    metrics.update(daily_rank_ic(
        stock_day, min_stocks=min_stocks_per_day, method=correlation_method
    ))
    metrics["stock_days"] = float(len(stock_day))
    return metrics, stock_day


def _selection_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    rank_ic = metrics.get("rank_ic_mean", float("nan"))
    oos_r2 = metrics.get("oos_r2_vs_historical_mean", float("nan"))
    return (
        rank_ic if np.isfinite(rank_ic) else -np.inf,
        oos_r2 if np.isfinite(oos_r2) else -np.inf,
        -metrics["mse"],
    )


def return_regressor_candidates(
    regressor: str, *, seed: int, stage: str = "coarse",
) -> list[dict[str, Any]]:
    """Return bounded candidate grids suitable for dense frozen embeddings."""
    if stage not in {"coarse", "fine"}:
        raise ValueError("stage must be coarse or fine")
    fine = stage == "fine"
    key = regressor.lower().replace("-", "_")
    if key == "elasticnet_sgd":
        return [
            {"alpha": alpha, "l1_ratio": l1_ratio, "seed": seed}
            for alpha in ((1e-7, 1e-6, 1e-5, 1e-4, 1e-3) if fine else (1e-6, 1e-5, 1e-4))
            for l1_ratio in ((0.02, 0.05, 0.2, 0.5, 0.8) if fine else (0.05, 0.2, 0.5))
        ]
    if key == "huber_sgd":
        return [
            {"alpha": alpha, "epsilon": epsilon, "seed": seed}
            for alpha in ((1e-7, 1e-6, 1e-5, 1e-4, 1e-3) if fine else (1e-6, 1e-5, 1e-4))
            for epsilon in ((0.005, 0.01, 0.02, 0.05, 0.1) if fine else (0.01, 0.05))
        ]
    if key == "small_mlp":
        return [
            {
                "hidden_layer_sizes": (hidden,), "alpha": alpha,
                "max_iter": max_iter, "seed": seed,
                "early_stopping": True,
                "early_stopping_patience": 10,
            }
            for hidden in ((32, 64, 128, 256) if fine else (32, 64))
            for alpha in ((1e-5, 1e-4, 1e-3, 1e-2) if fine else (1e-4, 1e-3))
            for max_iter in ((240,) if fine else (60, 120))
        ]
    raise ValueError("regressor must be one of: ridge, elasticnet_sgd, huber_sgd, small_mlp")


def make_return_regressor(regressor: str, params: Mapping[str, Any]) -> Any:
    """Construct one deterministic regressor from persisted parameters."""
    key = regressor.lower().replace("-", "_")
    values = dict(params)
    seed = int(values.pop("seed", 42))
    if key == "ridge":
        return Ridge(alpha=float(values["alpha"]))
    if key == "elasticnet_sgd":
        return SGDRegressor(
            loss="squared_error", penalty="elasticnet",
            alpha=float(values["alpha"]), l1_ratio=float(values["l1_ratio"]),
            max_iter=2000, tol=1e-5, random_state=seed,
            learning_rate="invscaling", average=True,
        )
    if key == "huber_sgd":
        return SGDRegressor(
            loss="huber", penalty="l2", alpha=float(values["alpha"]),
            epsilon=float(values["epsilon"]), max_iter=2000, tol=1e-5,
            random_state=seed, learning_rate="invscaling", average=True,
        )
    if key == "small_mlp":
        selected_epoch = int(values.pop("selected_epoch", values["max_iter"]))
        values.pop("early_stopping", None)
        values.pop("early_stopping_patience", None)
        return TargetScaledMLPRegressor(
            hidden_layer_sizes=tuple(values["hidden_layer_sizes"]),
            alpha=float(values["alpha"]), max_iter=selected_epoch,
            batch_size=values.get("batch_size", "auto"), seed=seed,
            learning_rate_init=float(values.get("learning_rate_init", 1e-3)),
        )
    raise ValueError("regressor must be one of: ridge, elasticnet_sgd, huber_sgd, small_mlp")


def select_return_regressor(
    regressor: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    validation_frame: pd.DataFrame,
    *,
    stock_column: str,
    date_column: str,
    target_column: str,
    min_stocks_per_day: int,
    seed: int = 42,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    selection_correlation: str = "spearman",
) -> RegressorSelection:
    """Select a bounded model grid using stock-day validation IC."""
    key = regressor.lower().replace("-", "_")
    if key == "ridge":
        raise ValueError("use select_ridge_alpha for boundary-aware Ridge selection")
    grid = list(candidates) if candidates is not None else return_regressor_candidates(key, seed=seed)
    if not grid:
        raise ValueError("regressor candidate grid cannot be empty")
    historical_mean = float(np.mean(y_fit))
    scored = []
    audit: list[dict[str, Any]] = []
    for raw_params in grid:
        params = dict(raw_params)
        params.setdefault("seed", seed)
        if key == "small_mlp":
            model, predictions, selected_epoch, epoch_history = fit_mlp_with_validation(
                x_fit, y_fit, x_validation, validation_frame,
                params=params, stock_column=stock_column, date_column=date_column,
                target_column=target_column, min_stocks_per_day=min_stocks_per_day,
                selection_correlation=selection_correlation,
            )
            params["selected_epoch"] = selected_epoch
        else:
            model = make_return_regressor(key, params)
            model.fit(x_fit, y_fit)
            predictions = np.asarray(model.predict(x_validation), dtype=float)
            epoch_history = []
        metrics, _ = evaluate_stock_day_predictions(
            validation_frame, predictions, historical_mean=historical_mean,
            stock_column=stock_column, date_column=date_column,
            target_column=target_column, min_stocks_per_day=min_stocks_per_day,
            correlation_method=selection_correlation,
        )
        scored.append((_selection_key(metrics), params, model, predictions, metrics))
        audit.append({
            "params": params, "selected_epoch": params.get("selected_epoch"),
            "early_stopping": key == "small_mlp",
            "epoch_history": epoch_history,
            **metrics,
        })
    best = max(scored, key=lambda value: value[0])
    return RegressorSelection(key, best[1], best[2], best[3], best[4], audit)


def select_ridge_alpha(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    validation_frame: pd.DataFrame,
    alphas: Sequence[float],
    *,
    stock_column: str,
    date_column: str,
    target_column: str,
    min_stocks_per_day: int,
    max_expansions: int = 2,
    selection_correlation: str = "spearman",
) -> AlphaSelection:
    """Select Ridge alpha on stock-day validation and expand upper-bound wins."""
    candidates = sorted(set(float(alpha) for alpha in alphas))
    if not candidates or candidates[0] <= 0:
        raise ValueError("alphas must contain positive values")
    if max_expansions < 0:
        raise ValueError("max_expansions cannot be negative")
    audit: list[dict[str, Any]] = []
    historical_mean = float(np.mean(y_fit))
    scored = []
    pending = candidates
    for expansion in range(max_expansions + 1):
        for alpha in pending:
            model = Ridge(alpha=alpha)
            model.fit(x_fit, y_fit)
            predicted = np.asarray(model.predict(x_validation), dtype=float)
            metrics, _ = evaluate_stock_day_predictions(
                validation_frame, predicted, historical_mean=historical_mean,
                stock_column=stock_column, date_column=date_column,
                target_column=target_column, min_stocks_per_day=min_stocks_per_day,
                correlation_method=selection_correlation,
            )
            scored.append((_selection_key(metrics), alpha, model, predicted, metrics))
            audit.append({"alpha": alpha, "expansion_round": expansion, **metrics})
        best = max(scored, key=lambda value: value[0])
        largest_evaluated = max(value[1] for value in scored)
        if best[1] != largest_evaluated or expansion == max_expansions:
            return AlphaSelection(best[1], best[2], best[3], best[4], audit)
        pending = [largest_evaluated * 10.0]
    raise RuntimeError("unreachable alpha selection state")
