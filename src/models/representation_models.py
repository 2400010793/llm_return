"""Prediction models operating on already-fitted text representations."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
from scipy import sparse
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from src.evaluation.prediction_metrics import regression_metrics
from src.evaluation.classification import evaluate_binary_classification


@dataclass
class RegressionResult:
    predictions: np.ndarray
    model: object
    metrics: dict[str, float]


@dataclass
class ClassificationResult:
    probabilities: np.ndarray
    labels: np.ndarray
    model: object
    metrics: dict[str, float]
    best_params: dict[str, Any] = field(default_factory=dict)


def _model(name: str, alpha: float, random_state: int):
    key = name.lower()
    if key == "ols":
        return LinearRegression()
    if key == "ridge":
        return Ridge(alpha=alpha)
    if key == "lasso":
        return Lasso(alpha=alpha, max_iter=5000)
    if key == "random_forest":
        return RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    if key == "nn":
        return make_pipeline(StandardScaler(with_mean=False), MLPRegressor(hidden_layer_sizes=(128, 32), early_stopping=True, max_iter=300, random_state=random_state))
    raise ValueError("model must be one of: ols, ridge, lasso, random_forest, nn")


def _classifier(name: str, random_state: int, **params: Any):
    key = name.lower().replace("-", "_")
    if key in {"logistic", "logistic_regression"}:
        defaults = {"max_iter": 1000, "class_weight": "balanced"}
        defaults.update(params)
        return LogisticRegression(random_state=random_state, **defaults)
    if key in {"random_forest", "rf"}:
        defaults = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2}
        defaults.update(params)
        return RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1, **defaults)
    if key in {"nn", "mlp"}:
        defaults = {"hidden_layer_sizes": (128, 32), "early_stopping": False, "max_iter": 300, "solver": "adam", "batch_size": "auto", "learning_rate_init": 1e-3}
        defaults.update(params)
        return make_pipeline(StandardScaler(with_mean=False), MLPClassifier(random_state=random_state, **defaults))
    raise ValueError("classifier must be one of: logistic, random_forest, nn")


def fit_representation_regressor(
    x_train: np.ndarray | sparse.spmatrix,
    y_train: Iterable[float],
    x_predict: np.ndarray | sparse.spmatrix,
    y_predict: Iterable[float],
    *,
    model_name: str = "ridge",
    alpha: float = 100.0,
    random_state: int = 42,
) -> tuple[np.ndarray, object, dict[str, float]]:
    """Fit one return model on a representation fitted by the caller."""
    y_train_array = np.asarray(list(y_train), dtype=float)
    y_predict_array = np.asarray(list(y_predict), dtype=float)
    finite = np.isfinite(y_train_array)
    if not finite.any():
        raise ValueError("training target has no finite values")
    model = _model(model_name, alpha, random_state)
    model.fit(x_train[finite], y_train_array[finite])
    predictions = np.asarray(model.predict(x_predict), dtype=float)
    metrics = regression_metrics(y_predict_array, predictions) if np.isfinite(y_predict_array).any() else {}
    return predictions, model, metrics


def fit_return_regressor(
    x_train: np.ndarray | sparse.spmatrix,
    y_train: Iterable[float],
    x_predict: np.ndarray | sparse.spmatrix,
    y_predict: Iterable[float] | None = None,
    *,
    model_name: str = "ridge",
    alpha: float = 100.0,
    random_state: int = 42,
) -> RegressionResult:
    """Fit one independent regression model for one return horizon.

    ``x_train`` must already be produced by a training-only representation
    fit.  ``y_predict`` is used only for out-of-sample evaluation.
    """
    predictions, model, metrics = fit_representation_regressor(
        x_train, y_train, x_predict,
        [] if y_predict is None else y_predict,
        model_name=model_name, alpha=alpha, random_state=random_state,
    )
    return RegressionResult(predictions, model, metrics)


def fit_representation_classifier(
    x_train: np.ndarray | sparse.spmatrix,
    train_three_day_returns: Iterable[float],
    x_predict: np.ndarray | sparse.spmatrix,
    predict_three_day_returns: Iterable[float] | None = None,
    *,
    model_name: str = "logistic",
    random_state: int = 42,
    model_params: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, LogisticRegression, np.ndarray]:
    """Fit logistic sentiment classification from training-window labels."""
    returns = np.asarray(list(train_three_day_returns), dtype=float)
    finite = np.isfinite(returns)
    labels = (returns[finite] > 0).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("training window needs both sentiment classes")
    model = _classifier(model_name, random_state, **dict(model_params or {}))
    model.fit(x_train[finite], labels)
    probabilities = model.predict_proba(x_predict)[:, 1]
    return probabilities, model, labels


def classification_metrics(
    actual_three_day_returns: Iterable[float],
    probabilities: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate probabilities against the paper's three-day weak labels."""
    returns = np.asarray(list(actual_three_day_returns), dtype=float)
    probs = np.asarray(list(probabilities), dtype=float)
    if returns.shape != probs.shape:
        raise ValueError("actual returns and probabilities must have equal length")
    mask = np.isfinite(returns) & np.isfinite(probs)
    if not mask.any():
        raise ValueError("no finite classification observations")
    actual = (returns[mask] > 0).astype(int)
    predicted = (probs[mask] >= threshold).astype(int)
    return evaluate_binary_classification(returns, probs, threshold=threshold)


def fit_sentiment_classifier(
    x_train: np.ndarray | sparse.spmatrix,
    train_three_day_returns: Iterable[float],
    x_predict: np.ndarray | sparse.spmatrix,
    predict_three_day_returns: Iterable[float] | None = None,
    *,
    model_name: str = "logistic",
    random_state: int = 42,
    model_params: Mapping[str, Any] | None = None,
    validation_fraction: float = 0.2,
    param_grid: Mapping[str, Iterable[Any]] | None = None,
) -> ClassificationResult:
    """Fit a training-only classifier, optionally selecting parameters chronologically.

    Hyperparameters are selected on the tail of the training window, not by
    random cross-validation. The winning model is then refit on all training
    rows. MLP uses Adam explicitly and stochastic models receive the seed.
    """
    returns = np.asarray(list(train_three_day_returns), dtype=float)
    finite = np.isfinite(returns)
    labels = (returns[finite] > 0).astype(int)
    if len(labels) < 4 or len(np.unique(labels)) < 2:
        raise ValueError("training window needs at least four observations and both sentiment classes")
    best_params: dict[str, Any] = dict(model_params or {})
    if param_grid:
        cut = int(len(labels) * (1.0 - validation_fraction))
        cut = min(max(cut, 2), len(labels) - 2)
        best_score = -np.inf
        for candidate in ParameterGrid(dict(param_grid)):
            candidate_model = _classifier(model_name, random_state, **candidate)
            candidate_model.fit(x_train[finite][:cut], labels[:cut])
            candidate_probabilities = candidate_model.predict_proba(x_train[finite][cut:])[:, 1]
            candidate_metrics = classification_metrics(returns[finite][cut:], candidate_probabilities)
            score = candidate_metrics["auc"] if np.isfinite(candidate_metrics["auc"]) else candidate_metrics["accuracy"]
            if score > best_score:
                best_score, best_params = score, dict(candidate)
    probabilities, model, labels = fit_representation_classifier(
        x_train, train_three_day_returns, x_predict,
        model_name=model_name, random_state=random_state, model_params=best_params,
    )
    metrics = {}
    if predict_three_day_returns is not None:
        metrics = classification_metrics(predict_three_day_returns, probabilities)
    return ClassificationResult(probabilities, labels, model, metrics, best_params)
