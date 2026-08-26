import numpy as np

from src.models.dimension_reduction import fit_reduce


def test_pca_is_fitted_only_on_training_rows():
    train = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]], dtype=np.float32)
    predict = np.array([[1000.0, -1000.0]], dtype=np.float32)
    result = fit_reduce(train, predict, method="pca", n_components=1, random_state=7)
    np.testing.assert_allclose(result.reducer.mean_, train.mean(axis=0))
    assert result.train.shape == (3, 1)
    assert result.predict.shape == (1, 1)


def test_randomized_pca_is_reproducible_for_fixed_seed():
    rng = np.random.default_rng(123)
    train = rng.normal(size=(40, 8)).astype(np.float32)
    predict = rng.normal(size=(5, 8)).astype(np.float32)
    first = fit_reduce(train, predict, method="pca", n_components=3, random_state=42)
    second = fit_reduce(train, predict, method="pca", n_components=3, random_state=42)
    np.testing.assert_allclose(first.train, second.train)
    np.testing.assert_allclose(first.predict, second.predict)