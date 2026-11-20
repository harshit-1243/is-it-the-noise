"""EXP-G (Module 2): sensitivity of DP-SGD utility to its hyperparameters at a fixed eps = 4.

Clip norm C and training length are varied one at a time; sigma is re-calibrated each time so
every run spends exactly eps = 4 (delta = 1e-5). This is a sensitivity analysis, not tuning:
picking the best cell on the test set would itself leak privacy (see report, threats to validity).

Usage: python src/exp_sensitivity.py --seeds 10 --out results/exp_sensitivity
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import clean, load_dataset, make_splits   # noqa: E402
from evaluate import Logger, evaluate, summarise          # noqa: E402
from exp_main import METRICS                              # noqa: E402
from models import DP_MLP, calibrate_noise, make_model    # noqa: E402

EPS, DELTA = 4.0, 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pima_diabetes.csv")
    ap.add_argument("--out", default="results/exp_sensitivity")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    df = clean(load_dataset(a.data))
    n = len(make_splits(df, 0)["target_in"][1])
    grid = [("clip", c, 30) for c in (0.1, 0.5, 1.0, 2.0, 5.0)] + [("epochs", 1.0, e) for e in (10, 60, 120)]
    cache = {}
    rows = []
    for seed in range(a.seeds):
        S = make_splits(df, seed)
        for what, C, ep in grid:
            q = DP_MLP["batch_size"] / n
            steps = int(np.ceil(ep / q))
            if (ep,) not in cache:
                cache[(ep,)] = calibrate_noise(EPS, q, steps, DELTA)
            sigma = cache[(ep,)]
            kw = {"noise_multiplier": sigma, "clip_norm": C, "epochs": ep}
            r = evaluate(lambda s, kw=kw: make_model("DP-MLP", s, **kw), S, seed)
            rows.append({"seed": seed, "vary": what, "clip": C, "epochs": ep, "sigma": sigma, **r})
            log(f"seed {seed} C={C:<4} epochs={ep:<4} sigma={sigma:.3f} acc={r['test_acc']:.3f} "
                f"f1={r['f1']:.3f} shadowAUC={r['shadow_auc']:.3f}")
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    summ = summarise(runs, ["vary", "clip", "epochs", "sigma"], METRICS)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    log("SUMMARY")
    for _, r in summ.iterrows():
        log(f"  C={r['clip']:<4} epochs={r['epochs']:<4} sigma={r['sigma']:.3f} acc={r['test_acc_mean']:.3f}±{r['test_acc_ci95']:.3f} "
            f"f1={r['f1_mean']:.3f} shadowAUC={r['shadow_auc_mean']:.3f}")
    log("Done.")


if __name__ == "__main__":
    main()
