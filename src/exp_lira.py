"""EXP-B (Module 1 audit, Module 2 check): LiRA on Pima with 64 models per configuration.

Each model trains on a random half (384 records) of all 768; every model takes a turn as the
target (leave-one-out), giving 64 x 768 = 49,152 member/non-member decisions per configuration,
enough to read TPR at 1% and 0.1% FPR.

Usage: python src/exp_lira.py --models 64 --out results/exp_lira
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import FEATURES, TARGET, clean, load_dataset    # noqa: E402
from evaluate import Logger                                      # noqa: E402
from lira import lira_metrics, lira_scores, train_population     # noqa: E402
from models import DP_MLP, calibrate_noise, compute_epsilon, make_model  # noqa: E402

DELTA = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pima_diabetes.csv")
    ap.add_argument("--out", default="results/exp_lira")
    ap.add_argument("--models", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    df = clean(load_dataset(a.data))
    X, y = df[FEATURES].reset_index(drop=True), df[TARGET].to_numpy()
    n_train = len(y) // 2
    q = DP_MLP["batch_size"] / n_train
    steps = int(np.ceil(DP_MLP["epochs"] / q))
    sig = {e: calibrate_noise(e, q, steps, DELTA) for e in (1.0, 4.0, 8.0)}
    log(f"LiRA: N={len(y)}, each model trains on {n_train}, K={a.models}; DP q={q:.4f} steps={steps}")
    for e, s in sig.items():
        log(f"  eps={e:g}: sigma={s:.3f} (achieved {compute_epsilon(q, s, steps, DELTA):.3f})")
    cfgs = [("LogReg", "LogReg", {}), ("RandomForest", "RandomForest", {}), ("MLP", "MLP", {}),
            ("MLP-L2", "MLP-L2", {}), ("MLP-matched", "MLP-matched", {}),
            ("DP-MLP clip-only", "DP-MLP", {"noise_multiplier": 0.0}),
            ("DP-MLP eps=8", "DP-MLP", {"noise_multiplier": sig[8.0]}),
            ("DP-MLP eps=4", "DP-MLP", {"noise_multiplier": sig[4.0]}),
            ("DP-MLP eps=1", "DP-MLP", {"noise_multiplier": sig[1.0]}),
            ("DP-LR eps=4", "DP-LR", {"noise_multiplier": sig[4.0]})]
    rows, curves = [], {}
    for label, kind, kw in cfgs:
        masks, phi = train_population(lambda s, k=kind, kw=kw: make_model(k, s, **kw), X, y, a.models, seed=1)
        sc, lb = lira_scores(masks, phi)
        m = lira_metrics(sc, lb)
        fpr, tpr, _ = roc_curve(lb, sc)
        grid = np.logspace(-4, 0, 200)
        curves[label] = np.interp(grid, fpr, tpr)
        rows.append({"model": label, **m})
        log(f"  {label:<18} AUC={m['lira_auc']:.3f}  TPR@1%FPR={m['lira_tpr_at_1pct']:.4f}  "
            f"TPR@0.1%FPR={m['lira_tpr_at_0.1pct']:.4f}  (n={m['n_scores']})")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    np.savez(os.path.join(a.out, "roc_curves.npz"), fpr_grid=np.logspace(-4, 0, 200),
             **{k.replace(" ", "_"): v for k, v in curves.items()})
    json.dump({"sigma": {str(k): v for k, v in sig.items()}, "q": q, "steps": steps}, open(os.path.join(a.out, "accounting.json"), "w"))
    log("Done.")


if __name__ == "__main__":
    main()
