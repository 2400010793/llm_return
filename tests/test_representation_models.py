import numpy as np

from src.models.representation_models import classification_metrics, fit_return_regressor, fit_sentiment_classifier
from src.evaluation.classification import evaluate_binary_classification, summarize_classification_stability
from src.evaluation.mcs import classification_loss_matrix, model_confidence_set


def test_return_regression_models_use_embedding_vectors() -> None:
    x_train = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y_train = [0.1, -0.1, 0.2, -0.2]
    for name in ["ridge", "lasso", "random_forest", "nn"]:
        result = fit_return_regressor(x_train, y_train, x_train[:2], [0.1, -0.1], model_name=name, alpha=0.01)
        assert result.predictions.shape == (2,)
        assert result.metrics["n"] == 2.0


def test_logistic_sentiment_classifier_uses_three_day_labels() -> None:
    x = np.array([[2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5]])
    result = fit_sentiment_classifier(x, [0.1, -0.1, 0.2, -0.2], x[:2], [0.1, -0.1])
    assert result.probabilities.shape == (2,)
    assert result.metrics["n"] == 2.0
    assert classification_metrics([0.1, -0.1], result.probabilities)["accuracy"] >= 0.5


def test_classification_seed_and_chronological_tuning_are_recorded() -> None:
    x = np.array([
        [2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5],
        [1.2, 0.0], [0.0, 1.2], [1.0, 0.0], [0.0, 1.0],
    ])
    returns = [0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2]
    first = fit_sentiment_classifier(
        x, returns, x[:2], returns[:2],
        model_name="mlp", random_state=7,
        param_grid={"hidden_layer_sizes": [(8,)], "solver": ["adam"], "max_iter": [80], "early_stopping": [False]},
    )
    second = fit_sentiment_classifier(
        x, returns, x[:2], returns[:2],
        model_name="mlp", random_state=7,
        param_grid={"hidden_layer_sizes": [(8,)], "solver": ["adam"], "max_iter": [80], "early_stopping": [False]},
    )
    assert first.best_params == {"hidden_layer_sizes": (8,), "solver": "adam", "max_iter": 80, "early_stopping": False}
    assert np.allclose(first.probabilities, second.probabilities)


def test_unified_classification_metrics_and_stability_summary() -> None:
    metrics = evaluate_binary_classification([0.1, -0.1, 0.2, -0.2], [0.9, 0.2, 0.8, 0.1])
    assert metrics["auc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["log_loss"] < 0.3
    summary = summarize_classification_stability([
        {"representation": "x", "classifier": "logistic", "tuned": False, "seed": 1, "auc": 0.6, "f1": 0.5, "accuracy": 0.5, "log_loss": 0.7, "brier": 0.2, "mcc": 0.0},
        {"representation": "x", "classifier": "logistic", "tuned": False, "seed": 2, "auc": 0.8, "f1": 0.7, "accuracy": 0.75, "log_loss": 0.5, "brier": 0.1, "mcc": 0.5},
    ])
    assert summary[0]["n_seeds"] == 2
    assert summary[0]["auc_mean"] == 0.7
    assert summary[0]["auc_std"] > 0


def test_mcs_uses_aligned_losses_and_reproducible_bootstrap() -> None:
    actual = [0.1, -0.1, 0.2, -0.2] * 8
    losses = classification_loss_matrix(
        actual,
        {
            "good": [0.9, 0.1, 0.8, 0.2] * 8,
            "weak": [0.6, 0.4, 0.55, 0.45] * 8,
            "bad": [0.1, 0.9, 0.2, 0.8] * 8,
        },
    )
    first = model_confidence_set(losses, bootstrap_reps=100, block_length=2, seed=9)
    second = model_confidence_set(losses, bootstrap_reps=100, block_length=2, seed=9)
    assert first == second
    assert first["n_observations"] == 32
    assert "good" in first["included_models"]
