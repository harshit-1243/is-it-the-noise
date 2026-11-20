"""Likelihood Ratio Attack (LiRA, Carlini et al., 2022), online variant — Module 1 privacy audit.

Train K models, each on a random half of the data pool, so every record is IN for about half
of them and OUT for the rest. For a target model t and record i, compare the target's
logit-scaled confidence phi_t(i) with two Gaussians fitted on the *other* models:
  IN  distribution: phi of models that trained on i
  OUT distribution: phi of models that did not.
Score = log N(phi; mu_in, sd_in) - log N(phi; mu_out, sd_out). Every model takes a turn as the
target (leave-one-out), and all scores are pooled into one ROC curve.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-6


def logit_confidence(model, X, y) -> np.ndarray:
    p1 = np.clip(model.predict_proba(X)[:, 1], EPS, 1 - EPS)
    p_true = np.where(y == 1, p1, 1 - p1)
    return np.log(p_true) - np.log(1 - p_true)


def train_population(make_fn, X, y, n_models: int, seed: int = 0):
    """Returns membership masks (K, N) and logit confidences (K, N)."""
    rng = np.random.default_rng(seed)
    N = len(y)
    masks = np.zeros((n_models, N), dtype=bool)
    phi = np.zeros((n_models, N))
    Xa = X.reset_index(drop=True) if hasattr(X, "reset_index") else X
    for k in range(n_models):
        idx = rng.permutation(N)[: N // 2]
        masks[k, idx] = True
        model = make_fn(seed * 1000 + k).fit(Xa.iloc[idx] if hasattr(Xa, "iloc") else Xa[idx], y[idx])
        phi[k] = logit_confidence(model, Xa, y)
    return masks, phi


def lira_scores(masks: np.ndarray, phi: np.ndarray, min_sd: float = 1e-2):
    """Leave-one-out online LiRA. Returns pooled (scores, labels)."""
    K, N = masks.shape
    scores, labels = [], []
    for t in range(K):
        others = np.arange(K) != t
        m_o, p_o = masks[others], phi[others]
        n_in = m_o.sum(0)
        n_out = (~m_o).sum(0)
        mu_in = np.where(n_in > 0, (p_o * m_o).sum(0) / np.maximum(n_in, 1), 0.0)
        mu_out = np.where(n_out > 0, (p_o * ~m_o).sum(0) / np.maximum(n_out, 1), 0.0)
        var_in = ((p_o - mu_in) ** 2 * m_o).sum(0) / np.maximum(n_in - 1, 1)
        var_out = ((p_o - mu_out) ** 2 * ~m_o).sum(0) / np.maximum(n_out - 1, 1)
        sd_in = np.sqrt(np.maximum(var_in, min_sd ** 2))
        sd_out = np.sqrt(np.maximum(var_out, min_sd ** 2))
        s = norm.logpdf(phi[t], mu_in, sd_in) - norm.logpdf(phi[t], mu_out, sd_out)
        scores.append(s)
        labels.append(masks[t])
    return np.concatenate(scores), np.concatenate(labels).astype(int)


def tpr_at(fpr, tpr, target):
    """Largest TPR achievable with FPR <= target (no interpolation: conservative)."""
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def lira_metrics(scores, labels) -> dict:
    fpr, tpr, _ = roc_curve(labels, scores)
    return {"lira_auc": float(roc_auc_score(labels, scores)),
            "lira_tpr_at_1pct": tpr_at(fpr, tpr, 0.01),
            "lira_tpr_at_0.1pct": tpr_at(fpr, tpr, 0.001),
            "lira_balanced_acc": float(np.max(1 - (fpr + (1 - tpr)) / 2)),
            "n_scores": int(len(labels))}
