import numpy as np

from src.models.text_representations import fit_precomputed, fit_representation


def test_classical_representations_are_train_fitted() -> None:
    result = fit_representation("tfidf", ["盈利增长", "风险上升"], ["盈利增长"])
    assert result.train.shape[0] == 2
    assert result.predict.shape[0] == 1


def test_lexicon_and_lmmd_are_numeric() -> None:
    result = fit_representation("lmmd", ["盈利增长"], ["风险下降"])
    assert result.train.shape == (1, len(result.train[0]))


def test_precomputed_embeddings_validate_dimension() -> None:
    result = fit_precomputed(np.ones((2, 4)), np.zeros((1, 4)), "qwen")
    assert result.train.shape == (2, 4)
