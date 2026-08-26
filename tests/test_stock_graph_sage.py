import joblib
import numpy as np

from src.models.stock_graph_sage import StockGraphSAGEClassifier, weighted_accuracy


def test_weighted_accuracy_uses_announcement_counts():
    labels = np.array([1, 0])
    probabilities = np.array([0.9, 0.9])
    weights = np.array([3, 1])
    assert weighted_accuracy(labels, probabilities, weights) == 0.75


def test_graphsage_fits_predicts_and_round_trips(tmp_path):
    rng = np.random.default_rng(7)
    rows = 160
    own = rng.normal(size=(rows, 4)).astype(np.float32)
    neighbor = rng.normal(size=(rows, 4)).astype(np.float32)
    labels = ((own[:, 0] + neighbor[:, 0]) > 0).astype(np.float32)
    features = np.concatenate((own, neighbor), axis=1)
    model = StockGraphSAGEClassifier(
        input_dimension=4,
        use_neighbors=True,
        hidden_size=16,
        dropout=0.0,
        learning_rate=0.01,
        max_epochs=12,
        patience=4,
        batch_size=32,
        random_state=42,
        num_threads=1,
    ).fit(
        features[:120],
        labels[:120],
        sample_weight=np.ones(120),
        validation_data=(features[120:], labels[120:], np.ones(40)),
    )
    probabilities = model.predict_proba(features[120:])
    assert probabilities.shape == (40, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert 1 <= model.best_epoch_ <= 12

    destination = tmp_path / "model.joblib"
    joblib.dump(model, destination)
    restored = joblib.load(destination)
    np.testing.assert_allclose(
        restored.predict_proba(features[120:]), probabilities, atol=1e-7
    )
