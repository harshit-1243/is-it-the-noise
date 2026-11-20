"""Black-box membership inference attacks (MIA).

Threat model: the attacker can query the deployed model and receive its
confidence scores; knows the model type and training recipe; and owns a
disjoint sample from the same population (the "shadow pool"). The attacker
does NOT see the target's training data or parameters.

1. Loss-threshold attack (Yeom et al., 2018): members tend to have lower loss.
2. Shadow-model attack (Shokri et al., 2017): train shadow models that imitate
   the target, label their in/out records, and learn an attack classifier.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

EPS = 1e-7


def per_example_loss(model, X, y) -> np.ndarray:
    p = np.clip(model.predict_proba(X)[:, 1], EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def attack_features(model, X, y) -> np.ndarray:
    """Features the attacker computes from black-box output: confidence on the true
    label, its logit-scaled version (as in LiRA), and the label itself."""
    p1 = np.clip(model.predict_proba(X)[:, 1], EPS, 1 - EPS)
    p_true = np.where(y == 1, p1, 1 - p1)
    return np.column_stack([p_true, np.log(p_true) - np.log(1 - p_true), y])


def leakage_metrics(scores_in: np.ndarray, scores_out: np.ndarray) -> dict:
    """Higher score = 'more likely a member'. Evaluated on balanced in/out sets."""
    y_true = np.r_[np.ones(len(scores_in)), np.zeros(len(scores_out))]
    s = np.r_[scores_in, scores_out]
    fpr, tpr, _ = roc_curve(y_true, s)
    return {
        "attack_auc": float(roc_auc_score(y_true, s)),
        "max_advantage": float(np.max(tpr - fpr)),             # Yeom et al. membership advantage
        "tpr_at_5fpr": float(np.interp(0.05, fpr, tpr)),
    }


def loss_threshold_attack(model, X_in, y_in, X_out, y_out) -> dict:
    s_in = -per_example_loss(model, X_in, y_in)
    s_out = -per_example_loss(model, X_out, y_out)
    return {"scores_in": s_in, "scores_out": s_out, **leakage_metrics(s_in, s_out)}


def shadow_model_attack(make_shadow, X_shadow, y_shadow, target, X_in, y_in, X_out, y_out,
                        n_shadows: int = 8, seed: int = 0) -> dict:
    """Train `n_shadows` shadow models on random halves of the shadow pool, build an
    in/out-labelled attack dataset, fit an attack classifier, then attack the target."""
    feats, labels = [], []
    for k in range(n_shadows):
        Xa, Xb, ya, yb = train_test_split(X_shadow, y_shadow, test_size=0.5,
                                          stratify=y_shadow, random_state=seed * 100 + k)
        shadow = make_shadow(seed * 100 + k).fit(Xa, ya)
        feats += [attack_features(shadow, Xa, ya), attack_features(shadow, Xb, yb)]
        labels += [np.ones(len(ya)), np.zeros(len(yb))]
    attack_clf = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=seed)
    attack_clf.fit(np.vstack(feats), np.concatenate(labels))

    s_in = attack_clf.predict_proba(attack_features(target, X_in, y_in))[:, 1]
    s_out = attack_clf.predict_proba(attack_features(target, X_out, y_out))[:, 1]
    tpr = float(np.mean(s_in >= 0.5))
    fpr = float(np.mean(s_out >= 0.5))
    return {
        "scores_in": s_in, "scores_out": s_out,
        **leakage_metrics(s_in, s_out),
        "attack_acc@0.5": 0.5 * (tpr + (1 - fpr)),              # realistic: threshold fixed in advance
        "adv@0.5": tpr - fpr,
    }
