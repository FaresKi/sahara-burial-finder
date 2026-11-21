"""Shared background (anomaly) models: one interface for texture features and embeddings.

``fit_score(X, method) -> scores`` fits an unsupervised background model on ALL rows of ``X``
(n_samples, n_features) and scores each row; higher means more anomalous. No labels involved.
Features are robust-scaled (median/IQR) first. Methods: ``iforest``, ``knn``, ``maha``.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

METHODS = ("iforest", "knn", "maha")


def fit_score(
    X: np.ndarray,
    method: str = "iforest",
    *,
    seed: int = 0,
    k: int = 10,
    n_estimators: int = 200,
) -> np.ndarray:
    """Anomaly score per row of ``X`` (float64 array, shape (n,)).

    - ``iforest``: negated IsolationForest ``score_samples`` (``n_estimators``, seeded).
    - ``knn``: mean Euclidean distance to the ``k`` nearest other rows (k capped at n-1).
    - ``maha``: Mahalanobis distance to a robust (MinCovDet) location/covariance; falls back to
      Ledoit-Wolf shrinkage when ``n_samples < 5 * n_features`` (e.g. 768-d embeddings) and to the
      empirical covariance if MinCovDet fails.
    """
    X = np.asarray(X, dtype="float64")
    if X.ndim != 2 or len(X) < 3:
        raise ValueError("X must be 2-D with at least 3 rows")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")
    Z = RobustScaler().fit_transform(X)
    if method == "iforest":
        model = IsolationForest(n_estimators=n_estimators, random_state=seed).fit(Z)
        return -model.score_samples(Z)
    if method == "knn":
        kk = min(k, len(Z) - 1)
        dist, _ = NearestNeighbors(n_neighbors=kk + 1).fit(Z).kneighbors(Z)
        return dist[:, 1:].mean(axis=1)
    if method == "maha":
        n, d = Z.shape
        if n < 5 * d:
            est = LedoitWolf().fit(Z)
        else:
            try:
                est = MinCovDet(random_state=seed).fit(Z)
            except ValueError:
                est = EmpiricalCovariance().fit(Z)
        return np.sqrt(np.maximum(est.mahalanobis(Z), 0))
    raise ValueError(f"unknown background method {method!r}; choose from {METHODS}")
