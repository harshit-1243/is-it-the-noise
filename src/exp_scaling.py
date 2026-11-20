"""EXP-F (Module 2): does DP get cheaper with more data?  CDC BRFSS 2015 Diabetes Health Indicators.

Train non-private and DP-SGD logistic regression on n = 200 ... 100,000 records and evaluate on a
fixed held-out set of 20,000. Features are min-max scaled with PUBLIC codebook ranges (no
data-dependent preprocessing, so the whole DP pipeline is private). Lot size grows with n.

Usage: python src/exp_scaling.py --seeds 3 --out results/exp_scaling
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import Logger, summarise                     # noqa: E402
from models import NumpyMLP, calibrate_noise, compute_epsilon  # noqa: E402

TARGET = "Diabetes_binary"
# public ranges from the BRFSS-derived codebook (binary columns are 0/1)
RANGES = {"BMI": (12, 98), "GenHlth": (1, 5), "MentHlth": (0, 30), "PhysHlth": (0, 30),
          "Age": (1, 13), "Education": (1, 6), "Income": (1, 8)}
SHA256 = "19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0"
N_GRID = [200, 1000, 5000, 25000, 100000]
EPS_GRID = [0.5, 1.0, 4.0]
DELTA_RULE = "delta = 1/(10 n)"


def load(path):
    import gzip
    raw = gzip.open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(path)
    return df, digest


def scale(df):
    X = df.drop(columns=[TARGET]).astype(float).copy()
    for c in X.columns:
        lo, hi = RANGES.get(c, (0, 1))
        X[c] = (X[c].clip(lo, hi) - lo) / (hi - lo)
    return X.to_numpy(), df[TARGET].to_numpy().astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/brfss2015_diabetes_binary.csv.gz")
    ap.add_argument("--out", default="results/exp_scaling")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    df, digest = load(a.data)
    prov = {"rows": int(len(df)), "columns": int(df.shape[1]), "positive_rate": float(df[TARGET].mean()),
            "duplicates": int(df.duplicated().sum()), "nulls": int(df.isna().sum().sum()),
            "sha256_uncompressed": digest, "sha256_matches_published": digest == SHA256}
    json.dump(prov, open(os.path.join(a.out, "provenance_check.json"), "w"), indent=2)
    log(f"BRFSS loaded: {prov}")
    X, y = scale(df)
    rows = []
    for seed in range(a.seeds):
        X_pool, X_te, y_pool, y_te = train_test_split(X, y, test_size=20000, stratify=y, random_state=seed)
        for n in N_GRID:
            Xn, _, yn, _ = train_test_split(X_pool, y_pool, train_size=n, stratify=y_pool, random_state=seed)
            delta = 1.0 / (10 * n)
            lot = int(min(max(32, n // 50), 2048))
            epochs = 20
            q = lot / n
            steps = int(np.ceil(epochs / q))
            # non-private reference
            lr = LogisticRegression(max_iter=3000).fit(Xn, yn)
            p = lr.predict_proba(X_te)[:, 1]
            rows.append({"seed": seed, "n": n, "model": "LogReg (non-private)", "epsilon": np.inf,
                         "roc_auc": roc_auc_score(y_te, p),
                         "bal_acc": balanced_accuracy_score(y_te, (p >= y.mean()).astype(int))})
            # same DP-SGD optimiser without noise: separates the optimiser's own gap from the cost of the noise
            m0 = NumpyMLP(hidden=0, epochs=epochs, lr=1.0, batch_size=lot, dp=True, clip_norm=1.0,
                          noise_multiplier=0.0, seed=seed).fit(Xn, yn)
            p0 = m0.predict_proba(X_te)[:, 1]
            rows.append({"seed": seed, "n": n, "model": "DP-LR clip-only (no noise)", "epsilon": np.inf,
                         "roc_auc": roc_auc_score(y_te, p0),
                         "bal_acc": balanced_accuracy_score(y_te, (p0 >= y.mean()).astype(int))})
            for eps in EPS_GRID:
                sigma = calibrate_noise(eps, q, steps, delta)
                m = NumpyMLP(hidden=0, epochs=epochs, lr=1.0, batch_size=lot, dp=True, clip_norm=1.0,
                             noise_multiplier=sigma, seed=seed).fit(Xn, yn)
                p = m.predict_proba(X_te)[:, 1]
                rows.append({"seed": seed, "n": n, "model": "DP-LR", "epsilon": eps, "sigma": sigma,
                             "achieved_eps": compute_epsilon(q, sigma, steps, delta), "delta": delta,
                             "lot": lot, "steps": steps,
                             "roc_auc": roc_auc_score(y_te, p),
                             "bal_acc": balanced_accuracy_score(y_te, (p >= y.mean()).astype(int))})
            last = [r for r in rows if r["seed"] == seed and r["n"] == n]
            log(f"seed {seed} n={n:<6} lot={lot:<5} " + "  ".join(
                f"{r['model'][:6]}{'' if np.isinf(r['epsilon']) else ' e=' + format(r['epsilon'], 'g')}: AUC={r['roc_auc']:.3f}"
                for r in last))
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    summ = summarise(runs, ["model", "epsilon", "n"], ["roc_auc", "bal_acc"])
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    log("SUMMARY (ROC-AUC mean ± 95% CI)")
    for _, r in summ.iterrows():
        log(f"  {r['model']:<22} eps={r['epsilon']:<4} n={r['n']:<6} AUC={r['roc_auc_mean']:.3f}±{r['roc_auc_ci95']:.3f} "
            f"balAcc={r['bal_acc_mean']:.3f}")
    log("Done.")


if __name__ == "__main__":
    main()
