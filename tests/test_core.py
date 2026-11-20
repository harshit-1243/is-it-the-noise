"""Correctness checks for every mechanism used in IA-2.  Run:  python -m pytest -q tests"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compressive import Projection, private_labels  # noqa: E402
from ldp import laplace_features, randomised_response, rr_keep_prob, rr_unbiased_mean, PUBLIC_RANGES  # noqa: E402
from lira import lira_metrics, lira_scores, train_population  # noqa: E402
from models import NumpyMLP, calibrate_noise, compute_epsilon  # noqa: E402
from ppdm import Mondrian  # noqa: E402


# --------------------------------------------------------------------------- #
# Module 2: DP-SGD building blocks
# --------------------------------------------------------------------------- #
def _flat_params(m):
    if m.hidden > 0:
        return np.concatenate([m.W1_.ravel(), m.b1_, m.w2_, [m.b2_]])
    return np.concatenate([m.w2_, [m.b2_]])


def _set_params(m, v):
    if m.hidden > 0:
        d, h = m.W1_.shape
        i = 0
        m.W1_ = v[i:i + d * h].reshape(d, h); i += d * h
        m.b1_ = v[i:i + h]; i += h
        m.w2_ = v[i:i + h]; i += h
        m.b2_ = v[i]
    else:
        d = m.w2_.shape[0]
        m.w2_, m.b2_ = v[:d], v[d]


@pytest.mark.parametrize("hidden", [0, 5])
def test_per_example_gradients_match_finite_differences(hidden):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(4, 3)), np.array([0.0, 1.0, 1.0, 0.0])
    m = NumpyMLP(hidden=hidden)
    m._init(3, rng)
    if hidden == 0:
        m.w2_ = rng.normal(size=3); m.b2_ = 0.3
    G = m._per_example_grads(X, y)
    theta = _flat_params(m).copy()

    def loss(v, i):
        _set_params(m, v)
        p = m._forward(X[i:i + 1])[2][0]
        return -(y[i] * np.log(p) + (1 - y[i]) * np.log(1 - p))

    for i in range(len(X)):
        num = np.zeros_like(theta)
        for j in range(len(theta)):
            e = np.zeros_like(theta); e[j] = 1e-6
            num[j] = (loss(theta + e, i) - loss(theta - e, i)) / 2e-6
        _set_params(m, theta.copy())
        assert np.allclose(G[i], num, atol=1e-5), f"example {i}"


def test_dp_sgd_update_is_bounded_by_clipping():
    """With noise off, every DP-SGD update must have norm <= C * |lot| / expected_lot."""
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(64, 4)) * 10, rng.integers(0, 2, 64).astype(float)
    record = []

    class Spy(NumpyMLP):
        def _per_example_grads(self, Xb, yb):
            self._last_b = len(Xb)
            return super()._per_example_grads(Xb, yb)

        def _apply_update(self, g):
            record.append((np.linalg.norm(g), getattr(self, "_last_b", 0)))
            self._last_b = 0
            super()._apply_update(g)

    C = 0.5
    Spy(hidden=8, epochs=3, lr=0.1, batch_size=16, dp=True, clip_norm=C, noise_multiplier=0.0, seed=0).fit(X, y)
    expected_lot = 16
    assert record and all(nrm <= C * b / expected_lot + 1e-9 for nrm, b in record)


def test_accountant_monotone_and_calibration():
    q, steps, delta = 32 / 192, 180, 1e-5
    eps = [compute_epsilon(q, s, steps, delta) for s in (1.0, 2.0, 4.0, 8.0)]
    assert all(a > b for a, b in zip(eps, eps[1:])), "epsilon must fall as noise grows"
    assert compute_epsilon(q, 0.0, steps, delta) == float("inf")
    for target in (0.5, 1.0, 4.0):
        s = calibrate_noise(target, q, steps, delta)
        got = compute_epsilon(q, s, steps, delta)
        assert got <= target + 1e-6 and got > target * 0.99


# --------------------------------------------------------------------------- #
# Module 3: local DP mechanisms
# --------------------------------------------------------------------------- #
def test_randomised_response_rate_and_unbiased_estimate():
    rng = np.random.default_rng(2)
    y = (rng.random(200_000) < 0.35).astype(int)
    for eps in (0.5, 1.0, 2.0):
        yn = randomised_response(y, eps, rng)
        assert abs(np.mean(yn == y) - rr_keep_prob(eps)) < 0.005
        p = rr_keep_prob(eps)
        sd = np.sqrt(0.25 / len(y)) / (2 * p - 1)                 # std of the unbiased estimator
        assert abs(rr_unbiased_mean(yn, eps) - y.mean()) < 4 * sd


def test_laplace_noise_scale():
    rng = np.random.default_rng(3)
    n = 100_000
    X = pd.DataFrame({c: np.full(n, (lo + hi) / 2) for c, (lo, hi) in PUBLIC_RANGES.items()})
    eps = 0.5
    Xp = laplace_features(X, eps, rng)
    for c, (lo, hi) in PUBLIC_RANGES.items():
        u = (Xp[c] - (lo + hi) / 2) / (hi - lo)
        assert abs(u.var() - 2 / eps ** 2) / (2 / eps ** 2) < 0.05     # Var[Laplace(b)] = 2 b^2
        assert abs(u.mean()) < 0.05


# --------------------------------------------------------------------------- #
# Module 4: Mondrian k-anonymity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [2, 5, 10, 25])
def test_mondrian_is_k_anonymous(k):
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.integers(0, 15, 400), rng.normal(size=(400, 4)), rng.normal(30, 6, 400),
                         rng.normal(size=400), rng.integers(21, 80, 400)]).astype(float)
    mon = Mondrian(k=k).fit(X)
    Xt = mon.transform(X)
    _, counts = np.unique(Xt[:, list(mon.qi_cols)], axis=0, return_counts=True)
    assert counts.min() >= k and mon.min_class_size_ >= k
    other = [c for c in range(X.shape[1]) if c not in mon.qi_cols]
    assert np.allclose(Xt[:, other], X[:, other])                # non-QI attributes untouched
    assert 0.0 <= mon.ncp(X) <= 1.0


# --------------------------------------------------------------------------- #
# Module 5: compressive projections
# --------------------------------------------------------------------------- #
def test_dca_finds_discriminant_direction_pca_finds_variance():
    rng = np.random.default_rng(5)
    n = 4000
    y = rng.integers(0, 2, n)
    X = np.zeros((n, 8))
    X[:, 0] = y * 2.0 + rng.normal(0, 0.5, n)        # small variance, separates classes
    X[:, 1] = rng.normal(0, 5.0, n)                    # large variance, useless
    X[:, 2:] = rng.normal(0, 1, (n, 6))
    X[:, 7] = rng.uniform(21, 80, n)                   # "Age" column
    X[:, 1] *= 1.0
    # make col 1 dominate variance after standardisation by correlating several columns with it
    X[:, 3] = X[:, 1] / 5 + rng.normal(0, 0.1, n)
    X[:, 4] = X[:, 1] / 5 + rng.normal(0, 0.1, n)
    pca = Projection("pca", m=1).fit(X, y)
    dca = Projection("dca", m=1).fit(X, y)
    assert np.argmax(np.abs(dca.W_[:, 0])) == 0
    assert np.argmax(np.abs(pca.W_[:, 0])) in (1, 3, 4)


def test_pdca_downweights_private_attribute():
    rng = np.random.default_rng(6)
    n = 4000
    X = rng.normal(size=(n, 8))
    X[:, 7] = rng.uniform(21, 80, n)
    priv = private_labels(X)
    y = ((X[:, 0] + 1.5 * (priv - 0.5) + rng.normal(0, 0.3, n)) > 0).astype(int)   # utility uses x0 AND age
    dca = Projection("dca", m=1).fit(X, y)
    pdca = Projection("pdca", m=1, beta=5.0).fit(X, y)
    w_d, w_p = np.abs(dca.W_[:, 0]) / np.abs(dca.W_[:, 0]).sum(), np.abs(pdca.W_[:, 0]) / np.abs(pdca.W_[:, 0]).sum()
    assert w_p[7] < w_d[7]                              # less weight on age
    assert w_p[0] > w_d[0]


# --------------------------------------------------------------------------- #
# Module 1: LiRA sanity checks
# --------------------------------------------------------------------------- #
def test_lira_detects_memorisation_and_not_a_constant_model():
    from sklearn.dummy import DummyClassifier
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(200, 3)))
    y = rng.integers(0, 2, 200)
    m1, p1 = train_population(lambda s: KNeighborsClassifier(n_neighbors=1), X, y, n_models=16, seed=0)
    hi = lira_metrics(*lira_scores(m1, p1))
    m2, p2 = train_population(lambda s: DummyClassifier(strategy="prior"), X, y, n_models=16, seed=0)
    lo = lira_metrics(*lira_scores(m2, p2))
    assert hi["lira_auc"] > 0.75
    assert abs(lo["lira_auc"] - 0.5) < 0.05
