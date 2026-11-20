"""Compressive privacy (Module 5): release a low-dimensional projection Z = X W instead of X.

* PCA  - unsupervised: keeps the directions of largest variance.
* DCA  - Discriminant Component Analysis (Kung, 2017): keeps the directions that best separate
         the UTILITY label (diabetes). Components are the leading generalised eigenvectors of
         S_B w = lambda (S + rho I) w, where S_B is the between-class scatter of the utility label
         and S the total scatter. For a binary label S_B has rank 1, so DCA has exactly one
         discriminant component (L - 1 = 1); we therefore use m = 1 for DCA and PDCA.
* PDCA - privacy-aware DCA (this project's variant): prefers directions that separate the
         utility label but NOT a private attribute (age group >= 35 years):
         (S_B^util - beta * S_B^priv) w = lambda (S + rho I) w.

The data owner fits the projection on its own training data (imputed, raw units; standardised
internally). The projection is treated as public: an attacker can project its own data with it.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

AGE_COL = 7          # position of Age (data_prep.FEATURES order)
PRIVATE_AGE = 35.0   # private attribute: age >= 35 years


def private_labels(X_raw) -> np.ndarray:
    return (np.asarray(X_raw, dtype=float)[:, AGE_COL] >= PRIVATE_AGE).astype(int)


def scatter_between(X, labels):
    mu = X.mean(0)
    S = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(labels):
        Xc = X[labels == c]
        d = (Xc.mean(0) - mu)[:, None]
        S += len(Xc) / len(X) * (d @ d.T)
    return S


class Projection(BaseEstimator, TransformerMixin):
    """Fit on imputed raw features; standardises internally, then projects to m dimensions."""

    def __init__(self, method: str = "pca", m: int = 2, beta: float = 1.0, rho: float = 1e-3, rho_b: float = 0.0):
        self.method = method
        self.m = m
        self.beta = beta
        self.rho = rho
        self.rho_b = rho_b

    def _std(self, X):
        return (np.asarray(X, dtype=float) - self.mu_) / self.sd_

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Xs = self._std(X)
        d = Xs.shape[1]
        S = Xs.T @ Xs / len(Xs)
        tr = np.trace(S) / d
        if self.method == "none":
            self.W_ = np.eye(d)
        elif self.method == "pca":
            vals, vecs = np.linalg.eigh(S)
            self.W_ = vecs[:, np.argsort(vals)[::-1][: self.m]]
        elif self.method in ("dca", "pdca"):
            A = scatter_between(Xs, np.asarray(y))
            if self.method == "pdca":
                A = A - self.beta * scatter_between(Xs, private_labels(X))
            vals, vecs = eigh(A + self.rho_b * tr * np.eye(d), S + self.rho * tr * np.eye(d))
            self.W_ = vecs[:, np.argsort(vals)[::-1][: self.m]]
            self.eigvals_ = np.sort(vals)[::-1]
        else:
            raise ValueError(self.method)
        return self

    def transform(self, X):
        return self._std(X) @ self.W_

    def reconstruct(self, Z):
        """Least-squares inverse (what a naive attacker gets back from Z), in standardised units."""
        return Z @ np.linalg.pinv(self.W_)
