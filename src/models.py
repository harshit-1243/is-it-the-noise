"""Target models, including a from-scratch NumPy MLP trained with SGD or DP-SGD.

DP-SGD (Abadi et al., 2016):
  1. Poisson-sample a lot (each record included with probability q).
  2. Compute the gradient of the loss for every example separately.
  3. Clip each per-example gradient to L2 norm <= C.
  4. Sum, add Gaussian noise N(0, (sigma*C)^2 I), divide by expected lot size q*N.
  5. Take a gradient step.
The privacy cost (epsilon, delta) of the whole run is tracked with an RDP accountant.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import dp_accounting
from dp_accounting import rdp

logging.getLogger("absl").setLevel(logging.ERROR)
RDP_ORDERS = [1.25, 1.5, 1.75] + list(range(2, 64)) + [80, 96, 128, 256]


# --------------------------------------------------------------------------- #
# Privacy accounting
# --------------------------------------------------------------------------- #
def compute_epsilon(q: float, noise_multiplier: float, steps: int, delta: float) -> float:
    """Epsilon spent by `steps` iterations of the Poisson-subsampled Gaussian mechanism."""
    if noise_multiplier == 0:
        return float("inf")
    acc = rdp.RdpAccountant(RDP_ORDERS)
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(noise_multiplier)), steps)
    acc.compose(event)
    return float(acc.get_epsilon(delta))


def calibrate_noise(target_eps: float, q: float, steps: int, delta: float) -> float:
    """Smallest noise multiplier (binary search) that keeps the run within target_eps."""
    lo, hi = 0.3, 64.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if compute_epsilon(q, mid, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------- #
# NumPy MLP with optional DP-SGD
# --------------------------------------------------------------------------- #
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class NumpyMLP(BaseEstimator, ClassifierMixin):
    """One-hidden-layer ReLU MLP for binary classification (hidden=0 -> logistic regression).

    dp=False : plain mini-batch SGD (non-private baseline)
    dp=True  : DP-SGD with per-example clipping (clip_norm) and Gaussian noise (noise_multiplier)
    """

    def __init__(self, hidden=64, epochs=300, lr=0.05, batch_size=32, l2=0.0,
                 dp=False, clip_norm=1.0, noise_multiplier=0.0, seed=0):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.l2 = l2
        self.dp = dp
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.seed = seed

    # ---- parameters ------------------------------------------------------- #
    def _init(self, d, rng):
        h = self.hidden
        if h > 0:
            self.W1_ = rng.normal(0, np.sqrt(2.0 / d), size=(d, h))
            self.b1_ = np.zeros(h)
            self.w2_ = rng.normal(0, np.sqrt(1.0 / h), size=h)
        else:
            self.W1_, self.b1_ = None, None
            self.w2_ = np.zeros(d)
        self.b2_ = 0.0

    def _forward(self, X):
        if self.hidden > 0:
            a = X @ self.W1_ + self.b1_
            H = np.maximum(a, 0.0)
        else:
            a, H = None, X
        z = H @ self.w2_ + self.b2_
        return a, H, _sigmoid(z)

    def _per_example_grads(self, X, y):
        """Gradients of binary cross-entropy for each example, flattened: shape (B, P)."""
        a, H, p = self._forward(X)
        dz = p - y                                  # (B,)
        g_w2 = dz[:, None] * H                      # (B, h)
        g_b2 = dz[:, None]                          # (B, 1)
        if self.hidden > 0:
            dH = dz[:, None] * self.w2_[None, :] * (a > 0)          # (B, h)
            g_W1 = X[:, :, None] * dH[:, None, :]                   # (B, d, h)
            return np.concatenate([g_W1.reshape(len(X), -1), dH, g_w2, g_b2], axis=1)
        return np.concatenate([g_w2, g_b2], axis=1)

    def _apply_update(self, g):
        if self.hidden > 0:
            d, h = self.W1_.shape
            i = 0
            gW1 = g[i:i + d * h].reshape(d, h); i += d * h
            gb1 = g[i:i + h]; i += h
            gw2 = g[i:i + h]; i += h
            gb2 = g[i]
            self.W1_ -= self.lr * (gW1 + self.l2 * self.W1_)
            self.b1_ -= self.lr * gb1
        else:
            d = self.w2_.shape[0]
            gw2, gb2 = g[:d], g[d]
        self.w2_ -= self.lr * (gw2 + self.l2 * self.w2_)
        self.b2_ -= self.lr * gb2

    def _n_params(self, d):
        h = self.hidden
        return d * h + h + h + 1 if h > 0 else d + 1

    # ---- sklearn API ------------------------------------------------------- #
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self._init(d, rng)
        self.classes_ = np.array([0, 1])
        if not self.dp:
            for _ in range(self.epochs):
                perm = rng.permutation(n)
                for s in range(0, n, self.batch_size):
                    idx = perm[s:s + self.batch_size]
                    self._apply_update(self._per_example_grads(X[idx], y[idx]).mean(axis=0))
            self.steps_, self.sample_rate_ = None, None
            return self
        # ---------------- DP-SGD ---------------- #
        q = self.batch_size / n
        steps = int(np.ceil(self.epochs / q))
        P = self._n_params(d)
        C, sigma = self.clip_norm, self.noise_multiplier
        expected_lot = q * n
        for _ in range(steps):
            idx = np.where(rng.random(n) < q)[0]                 # Poisson sampling
            g_sum = np.zeros(P)
            if len(idx):
                G = self._per_example_grads(X[idx], y[idx])
                norms = np.linalg.norm(G, axis=1, keepdims=True)
                G = G / np.maximum(1.0, norms / C)                # per-example clipping
                g_sum = G.sum(axis=0)
            noise = rng.normal(0.0, sigma * C, size=P)            # Gaussian mechanism
            self._apply_update((g_sum + noise) / expected_lot)
        self.steps_, self.sample_rate_ = steps, q
        return self

    def predict_proba(self, X):
        p = self._forward(np.asarray(X, dtype=float))[2]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
# Model factory (each model is a full pipeline: impute -> scale -> classifier)
# --------------------------------------------------------------------------- #
NONPRIVATE_MLP = dict(hidden=64, epochs=1000, lr=0.05, batch_size=32)  # trained to fit, no regularisation (worst case)
DP_MLP = dict(hidden=64, epochs=30, lr=0.5, batch_size=32, clip_norm=1.0)
# IA-2 baselines that isolate *why* DP-SGD leaks less (see report, RQ2)
MATCHED_MLP = dict(hidden=64, epochs=30, lr=0.5, batch_size=32)          # DP-SGD recipe minus clipping and noise
L2_MLP = dict(hidden=64, epochs=1000, lr=0.05, batch_size=32, l2=1e-2)  # classic regularisation, no DP
DP_LR = dict(hidden=0, epochs=30, lr=0.5, batch_size=32, clip_norm=1.0)  # logistic regression trained with DP-SGD


def _classifier(name: str, seed: int, **kw):
    if name == "LogReg":
        return LogisticRegression(C=1.0, max_iter=2000)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1)
    if name == "MLP":
        return NumpyMLP(**{**NONPRIVATE_MLP, **kw}, seed=seed)
    if name == "MLP-matched":
        return NumpyMLP(**{**MATCHED_MLP, **kw}, seed=seed)
    if name == "MLP-L2":
        return NumpyMLP(**{**L2_MLP, **kw}, seed=seed)
    if name == "DP-MLP":
        return NumpyMLP(**{**DP_MLP, **kw}, dp=True, seed=seed)
    if name == "DP-LR":
        return NumpyMLP(**{**DP_LR, **kw}, dp=True, seed=seed)
    raise ValueError(name)


def make_model(name: str, seed: int, extra_steps=None, after_scale_steps=None, **kw) -> Pipeline:
    """impute -> [extra_steps] -> scale -> [after_scale_steps] -> classifier.

    extra_steps run on imputed raw-unit features (e.g. Mondrian k-anonymisation, Module 4);
    after_scale_steps run on standardised features (e.g. PCA/DCA projection, Module 5)."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    steps += list(extra_steps or [])
    steps += [("scale", StandardScaler())]
    steps += list(after_scale_steps or [])
    steps += [("clf", _classifier(name, seed, **kw))]
    return Pipeline(steps)
