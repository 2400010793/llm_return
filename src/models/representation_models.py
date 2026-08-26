"""Prediction models operating on already-fitted text representations."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
from scipy import sparse
from dataclasses import dataclass, field

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
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


class TemporalEarlyStoppingMLPClassifier(BaseEstimator, ClassifierMixin):
    """Dense MLP whose epoch budget is selected on a chronological fold."""

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (64,),
        alpha: float = 1e-3,
        max_iter: int = 120,
        batch_size: int | str = 256,
        learning_rate_init: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.batch_size = batch_size
        self.learning_rate_init = float(learning_rate_init)
        self.random_state = int(random_state)

    def _new_network(self, max_iter: int, *, one_epoch: bool = False) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=1 if one_epoch else int(max_iter),
            batch_size=self.batch_size,
            solver="adam",
            learning_rate_init=self.learning_rate_init,
            early_stopping=False,
            tol=0.0,
            n_iter_no_change=max(2, int(max_iter) + 1),
            random_state=self.random_state,
            warm_start=bool(one_epoch),
        )

    def fit(self, X: np.ndarray | sparse.spmatrix, y: np.ndarray) -> "TemporalEarlyStoppingMLPClassifier":
        labels = np.asarray(y, dtype=np.int8)
        if labels.ndim != 1 or len(labels) != len(X) or not np.isin(labels, (0, 1)).all():
            raise ValueError("y must be a binary vector aligned with X")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.scaler_ = StandardScaler(with_mean=False).fit(X)
        transformed = self.scaler_.transform(X)
        self.network_ = self._new_network(self.max_iter)
        self.network_.fit(transformed, labels)
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.n_iter_ = int(self.network_.n_iter_)
        self.loss_curve_ = list(self.network_.loss_curve_)
        self.selected_epoch_ = self.n_iter_
        return self

    @classmethod
    def from_fitted(
        cls,
        network: MLPClassifier,
        scaler: StandardScaler,
        *,
        hidden_layer_sizes: tuple[int, ...],
        alpha: float,
        max_iter: int,
        batch_size: int | str,
        learning_rate_init: float,
        random_state: int,
        selected_epoch: int,
    ) -> "TemporalEarlyStoppingMLPClassifier":
        model = cls(
            hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
            max_iter=max_iter, batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            random_state=random_state,
        )
        model.network_ = network
        model.scaler_ = scaler
        model.classes_ = np.array([0, 1], dtype=np.int8)
        model.n_iter_ = int(selected_epoch)
        model.selected_epoch_ = int(selected_epoch)
        model.loss_curve_ = list(getattr(network, "loss_curve_", []))
        return model

    def predict_proba(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        if not hasattr(self, "network_"):
            raise RuntimeError("model must be fitted before predict_proba")
        return self.network_.predict_proba(self.scaler_.transform(X))

    def predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(np.int8)


class _NBSVMClassifier(BaseEstimator, ClassifierMixin):
    """NB-SVM for sparse bag-of-words or dense feature matrices."""

    def __init__(self, C: float = 1.0, alpha: float = 1.0, class_weight: str | None = "balanced"):
        self.C = C
        self.alpha = alpha
        self.class_weight = class_weight

    def fit(self, X: np.ndarray | sparse.spmatrix, y: np.ndarray) -> "_NBSVMClassifier":
        y = np.asarray(y, dtype=int)
        if sparse.issparse(X):
            binary = X.tocsr(copy=True)
            binary.data = np.ones_like(binary.data)
            positive = binary[y == 1].sum(axis=0)
            negative = binary[y == 0].sum(axis=0)
        else:
            binary = (np.asarray(X) != 0).astype(np.float64)
            positive = binary[y == 1].sum(axis=0)
            negative = binary[y == 0].sum(axis=0)
        positive = np.asarray(positive).ravel()
        negative = np.asarray(negative).ravel()
        self.ratio_ = np.log((positive + self.alpha) / (positive.sum() + self.alpha * len(positive)))
        self.ratio_ -= np.log((negative + self.alpha) / (negative.sum() + self.alpha * len(negative)))
        self.model_ = LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=1000)
        self.model_.fit(self._weight(X), y)
        self.classes_ = self.model_.classes_
        return self

    def _weight(self, X: np.ndarray | sparse.spmatrix):
        if sparse.issparse(X):
            return X.multiply(self.ratio_)
        return np.asarray(X) * self.ratio_

    def predict_proba(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        return self.model_.predict_proba(self._weight(X))

    def predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        return self.model_.predict(self._weight(X))


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
    if key in {"linear_svm", "linear_svc", "svm"}:
        defaults = {"C": 1.0, "class_weight": "balanced"}
        defaults.update(params)
        return CalibratedClassifierCV(
            LinearSVC(random_state=random_state, **defaults), cv=2, method="sigmoid"
        )
    if key in {"nb_svm", "nbsvm"}:
        defaults = {"C": 1.0, "alpha": 1.0, "class_weight": "balanced"}
        defaults.update(params)
        return _NBSVMClassifier(**defaults)
    if key in {"knn", "k_nearest_neighbors", "k_neighbors"}:
        defaults = {"n_neighbors": 15, "weights": "distance", "metric": "cosine", "n_jobs": -1}
        defaults.update(params)
        return KNeighborsClassifier(**defaults)
    if key in {"multinomial_nb", "multinomial_naive_bayes", "mnb"}:
        defaults = {"alpha": 1.0, "fit_prior": True}
        defaults.update(params)
        return MultinomialNB(**defaults)
    if key in {"complement_nb", "complement_naive_bayes", "cnb"}:
        defaults = {"alpha": 1.0, "fit_prior": True, "norm": False}
        defaults.update(params)
        return ComplementNB(**defaults)
    if key in {"sgd", "sgd_logistic", "sgd_hinge"}:
        defaults = {"loss": "log_loss", "alpha": 1e-5, "max_iter": 1000, "class_weight": "balanced", "early_stopping": False}
        if key == "sgd_hinge":
            defaults["loss"] = "hinge"
        defaults.update(params)
        return SGDClassifier(random_state=random_state, **defaults)
    if key in {"random_forest", "rf"}:
        defaults = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2}
        defaults.update(params)
        return RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1, **defaults)
    if key in {"extra_trees", "extremely_randomized_trees", "et"}:
        defaults = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
        }
        defaults.update(params)
        return ExtraTreesClassifier(random_state=random_state, n_jobs=-1, **defaults)
    if key in {"hist_gradient_boosting", "hist_gbdt", "hgbt"}:
        defaults = {
            "learning_rate": 0.05,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.1,
            "early_stopping": True,
            "class_weight": "balanced",
        }
        defaults.update(params)
        return HistGradientBoostingClassifier(random_state=random_state, **defaults)
    if key in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "xgboost classifier requires requirements-boosting.txt"
            ) from error
        defaults = {
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "n_estimators": 400,
            "learning_rate": 0.05, "max_depth": 5,
            "min_child_weight": 5.0, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_lambda": 1.0,
            "n_jobs": -1, "verbosity": 0,
        }
        defaults.update(params)
        return XGBClassifier(random_state=random_state, **defaults)
    if key in {"lightgbm", "lgbm", "lgb"}:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError(
                "lightgbm classifier requires requirements-boosting.txt"
            ) from error
        defaults = {
            "objective": "binary", "n_estimators": 500,
            "learning_rate": 0.03, "num_leaves": 31,
            "max_depth": -1, "min_child_samples": 100,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_lambda": 1.0, "class_weight": "balanced",
            "n_jobs": -1, "verbosity": -1,
        }
        defaults.update(params)
        return LGBMClassifier(random_state=random_state, **defaults)
    if key in {"catboost", "cat"}:
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise ImportError(
                "catboost classifier requires requirements-boosting.txt"
            ) from error
        defaults = {
            "iterations": 500, "learning_rate": 0.03, "depth": 6,
            "loss_function": "Logloss", "auto_class_weights": "Balanced",
            "verbose": False, "thread_count": -1,
        }
        defaults.update(params)
        return CatBoostClassifier(random_seed=random_state, **defaults)
    if key == "lstm":
        from src.models.lstm_classifier import LSTMClassifier

        defaults = {
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 5,
            "batch_size": 512,
            "gradient_clip_norm": 1.0,
            "device": "auto",
        }
        defaults.update(params)
        return LSTMClassifier(random_state=random_state, **defaults)
    if key in {"simple_mlp", "small_mlp"}:
        defaults = {
            "hidden_layer_sizes": (64,),
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
            "max_iter": 120,
            "solver": "adam",
            "batch_size": 256,
            "learning_rate_init": 1e-3,
        }
        defaults.update(params)
        return make_pipeline(
            StandardScaler(with_mean=False),
            MLPClassifier(random_state=random_state, **defaults),
        )
    if key in {"nn", "mlp"}:
        defaults = {"hidden_layer_sizes": (128, 32), "early_stopping": False, "max_iter": 300, "solver": "adam", "batch_size": "auto", "learning_rate_init": 1e-3}
        defaults.update(params)
        return make_pipeline(StandardScaler(with_mean=False), MLPClassifier(random_state=random_state, **defaults))
    raise ValueError(
        "classifier must be one of: logistic, linear_svm, nb_svm, knn, "
        "random_forest, extra_trees, hist_gradient_boosting, xgboost, "
        "lightgbm, catboost, lstm, simple_mlp, nn"
    )


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
