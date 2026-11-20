"""Shared evaluation used by every IA-2 experiment: utility + two membership attacks + CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from attacks import loss_threshold_attack, shadow_model_attack


def utility(model, X_in, y_in, X_out, y_out) -> dict:
    p = model.predict_proba(X_out)[:, 1]
    yhat = (p >= 0.5).astype(int)
    train_acc = float(np.mean(model.predict(X_in) == y_in))
    test_acc = float(np.mean(yhat == y_out))
    return {
        "train_acc": train_acc, "test_acc": test_acc, "gen_gap": train_acc - test_acc,
        "precision": precision_score(y_out, yhat, zero_division=0),
        "recall": recall_score(y_out, yhat, zero_division=0),
        "f1": f1_score(y_out, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y_out, p),
    }


def evaluate(make_fn, S: dict, seed: int, n_shadows: int = 8) -> dict:
    """Train target with make_fn(seed) on members, score utility on non-members, run both MIAs.
    make_fn(s) must build the *whole* recipe (the attacker knows it and replays it for shadows)."""
    X_sh, y_sh = S["shadow"]; X_in, y_in = S["target_in"]; X_out, y_out = S["target_out"]
    target = make_fn(seed).fit(X_in, y_in)
    u = utility(target, X_in, y_in, X_out, y_out)
    la = loss_threshold_attack(target, X_in, y_in, X_out, y_out)
    sa = shadow_model_attack(make_fn, X_sh, y_sh, target, X_in, y_in, X_out, y_out,
                             n_shadows=n_shadows, seed=seed)
    return {**u,
            "loss_auc": la["attack_auc"], "loss_adv": la["max_advantage"], "loss_tpr5": la["tpr_at_5fpr"],
            "shadow_auc": sa["attack_auc"], "shadow_acc": sa["attack_acc@0.5"], "shadow_adv": sa["adv@0.5"]}


def ci95(x) -> tuple[float, float, float]:
    """Mean and 95% t-interval half-width over seeds."""
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    if len(x) < 2:
        return m, float("nan"), float("nan")
    h = float(stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x)))
    return m, h, float(np.std(x, ddof=1))


def summarise(runs: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in runs.groupby(by, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["n_seeds"] = len(g)
        for m in metrics:
            mean, half, sd = ci95(g[m])
            r[f"{m}_mean"], r[f"{m}_ci95"], r[f"{m}_sd"] = mean, half, sd
        rows.append(r)
    return pd.DataFrame(rows)


def paired_test(runs: pd.DataFrame, a: str, b: str, metric: str, key="model") -> dict:
    """Wilcoxon signed-rank test over seeds for configuration a vs b (paired by seed)."""
    A = runs[runs[key] == a].set_index("seed")[metric]
    B = runs[runs[key] == b].set_index("seed")[metric]
    idx = A.index.intersection(B.index)
    d = (A.loc[idx] - B.loc[idx]).to_numpy()
    p = float(stats.wilcoxon(A.loc[idx], B.loc[idx]).pvalue) if np.any(d != 0) else 1.0
    return {"a": a, "b": b, "metric": metric, "mean_a": float(A.loc[idx].mean()),
            "mean_b": float(B.loc[idx].mean()), "mean_diff": float(d.mean()), "p_wilcoxon": p, "n": len(idx)}


class Logger:
    """Timestamped console + file log (the log files are part of the evidence)."""

    def __init__(self, path):
        import time
        self._t = time.time
        self.t0 = time.time()
        self.f = open(path, "w")

    def __call__(self, msg=""):
        line = f"[{self._t() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()
