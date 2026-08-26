import numpy as np

from scripts.run_pooled_embedding_classification import dense_grid
from src.models.lstm_classifier import LSTMClassifier
from src.models.representation_models import _classifier


def test_lstm_classifier_has_sklearn_probability_interface() -> None:
    rng = np.random.default_rng(12)
    features = rng.normal(size=(24, 6)).astype(np.float32)
    labels = (features[:, 0] + 0.25 * features[:, 1] > 0).astype(np.int8)

    model = LSTMClassifier(
        hidden_size=8,
        epochs=2,
        batch_size=8,
        device="cpu",
        random_state=7,
    ).fit(features, labels)
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (24, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert model.sequence_length_ == 1
    assert model.n_features_in_ == 6
    assert np.array_equal(model.predict(features), (probabilities[:, 1] >= 0.5).astype(np.int8))


def test_lstm_is_registered_with_the_pooled_classifier_runner() -> None:
    model = _classifier("lstm", 42, epochs=1, batch_size=4, device="cpu")
    assert isinstance(model, LSTMClassifier)
    candidate = dense_grid("lstm", "coarse")
    assert candidate["hidden_size"] == [64]
    assert candidate["epochs"] == [5]