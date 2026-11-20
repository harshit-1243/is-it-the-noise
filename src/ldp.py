"""Local differential privacy (Module 3): each patient perturbs their own record before sharing.

* Randomised response (RR) for the binary diabetes label: report the true label with
  probability e^eps / (1 + e^eps), otherwise the flipped label. This is eps-LDP.
* Laplace mechanism for each numeric feature: clip to a PUBLIC clinical range [lo, hi],
  rescale to [0, 1] (sensitivity 1), add Laplace(0, 1/eps_j). This is eps_j-LDP per feature.
* Sequential composition: a record perturbed attribute-by-attribute with eps_j is
  (sum_j eps_j)-LDP. We split the total budget equally across the 9 attributes.

The hospital never sees true values; it trains on the noisy reports. Predictions are made on
clean inputs (the clinic measures the patient at prediction time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Public, data-independent ranges (clinical plausibility), so the perturbation leaks nothing.
PUBLIC_RANGES = {
    "Pregnancies": (0, 17), "Glucose": (40, 200), "BloodPressure": (20, 125),
    "SkinThickness": (5, 100), "Insulin": (10, 850), "BMI": (15, 70),
    "DiabetesPedigreeFunction": (0.05, 2.5), "Age": (21, 81),
}


def rr_keep_prob(eps: float) -> float:
    return float(np.exp(eps) / (1.0 + np.exp(eps)))


def randomised_response(y: np.ndarray, eps: float, rng) -> np.ndarray:
    keep = rng.random(len(y)) < rr_keep_prob(eps)
    return np.where(keep, y, 1 - y)


def rr_unbiased_mean(y_noisy: np.ndarray, eps: float) -> float:
    """Unbiased estimate of the true positive rate from RR reports (frequency estimation)."""
    p = rr_keep_prob(eps)
    return float((y_noisy.mean() - (1 - p)) / (2 * p - 1))


def laplace_features(X: pd.DataFrame, eps_per_feature: float, rng) -> pd.DataFrame:
    out = X.copy()
    for col, (lo, hi) in PUBLIC_RANGES.items():
        v = out[col].to_numpy(dtype=float)
        v = np.where(np.isnan(v), (lo + hi) / 2.0, v)          # missing -> public midpoint
        u = (np.clip(v, lo, hi) - lo) / (hi - lo)               # [0, 1], sensitivity 1
        u = u + rng.laplace(0.0, 1.0 / eps_per_feature, size=len(u))
        out[col] = lo + u * (hi - lo)                           # back to clinical units (unclipped: unbiased)
    return out


class LDPTrained(BaseEstimator, ClassifierMixin):
    """Wraps any pipeline: fit() perturbs the training records locally, predict() uses clean inputs.

    mode='label' : only the diagnosis is randomised (all eps on the label)
    mode='full'  : features + label, eps split equally over 9 attributes
    """

    def __init__(self, base, eps: float, mode: str = "label", seed: int = 0):
        self.base = base
        self.eps = eps
        self.mode = mode
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed + 7919)
        y = np.asarray(y)
        if self.mode == "label":
            Xp, yp = X, randomised_response(y, self.eps, rng)
        elif self.mode == "full":
            e = self.eps / (len(PUBLIC_RANGES) + 1)
            Xp, yp = laplace_features(X, e, rng), randomised_response(y, e, rng)
        else:
            raise ValueError(self.mode)
        if len(np.unique(yp)) < 2:                              # degenerate draw: keep training defined
            yp = yp.copy(); yp[0] = 1 - yp[0]
        self.model_ = clone(self.base).fit(Xp, yp)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)
