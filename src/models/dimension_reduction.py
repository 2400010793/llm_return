"""Leakage-safe dimensionality reduction for text representations.

Dense embeddings can use PCA. Sparse TF-IDF/BOW matrices should use
TruncatedSVD (latent semantic analysis) instead of centering the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD


@dataclass
class ReductionResult:
    train: np.ndarray
    predict: np.ndarray
    reducer: PCA | TruncatedSVD


def fit_reduce(
    x_train: np.ndarray | sparse.spmatrix,
    x_predict: np.ndarray | sparse.spmatrix,
    *,
    n_components: int = 100,
    method: str = "auto",
    random_state: int = 42,
) -> ReductionResult:
    """Fit a reducer on training rows only and transform prediction rows."""
    if n_components < 1:
        raise ValueError("n_components must be positive")
    key = method.lower().replace("-", "_")
    if key == "auto":
        key = "svd" if sparse.issparse(x_train) else "pca"
    if key in {"svd", "truncated_svd", "lsa"}:
        if not sparse.issparse(x_train):
            x_train = sparse.csr_matrix(np.asarray(x_train))
            x_predict = sparse.csr_matrix(np.asarray(x_predict))
        limit = min(x_train.shape[0], x_train.shape[1])
        if n_components >= limit:
            raise ValueError(f"n_components must be smaller than min(train shape)={limit} for TruncatedSVD")
        reducer: PCA | TruncatedSVD = TruncatedSVD(n_components=n_components, random_state=random_state)
    elif key == "pca":
        if sparse.issparse(x_train):
            raise ValueError("PCA requires dense input; use method='svd' for sparse text matrices")
        x_train = np.asarray(x_train, dtype=np.float32)
        x_predict = np.asarray(x_predict, dtype=np.float32)
        limit = min(x_train.shape[0], x_train.shape[1])
        if n_components > limit:
            raise ValueError(f"n_components cannot exceed min(train shape)={limit} for PCA")
        # Dense transformer matrices can contain hundreds of thousands of
        # rows. Randomized PCA avoids the much more expensive full SVD while
        # remaining deterministic under ``random_state``.
        reducer = PCA(
            n_components=n_components,
            svd_solver="randomized" if n_components < limit else "full",
            random_state=random_state,
        )
    else:
        raise ValueError("method must be one of: auto, pca, svd, lsa")
    return ReductionResult(
        np.asarray(reducer.fit_transform(x_train), dtype=np.float32),
        np.asarray(reducer.transform(x_predict), dtype=np.float32),
        reducer,
    )
