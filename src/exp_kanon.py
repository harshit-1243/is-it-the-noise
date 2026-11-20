"""EXP-D (Module 4): does k-anonymising the training data (Mondrian on Age, Pregnancies, BMI)
reduce membership leakage of the trained model, and what does it cost in utility?

Usage: python src/exp_kanon.py --seeds 10 --out results/exp_kanon
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import clean, load_dataset, make_splits   # noqa: E402
from evaluate import Logger, evaluate, summarise          # noqa: E402
from exp_main import METRICS                              # noqa: E402
from models import make_model                             # noqa: E402
from ppdm import Mondrian                                 # noqa: E402

K_GRID = [1, 2, 5, 10, 25, 50]   # k = 1 means no anonymisation


def build(kind, k):
    def fn(s):
        extra = [("anon", Mondrian(k=k))] if k > 1 else None
        return make_model(kind, s, extra_steps=extra)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pima_diabetes.csv")
    ap.add_argument("--out", default="results/exp_kanon")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    df = clean(load_dataset(a.data))
    rows = []
    for seed in range(a.seeds):
        S = make_splits(df, seed)
        X_in, y_in = S["target_in"]
        log(f"--- seed {seed}")
        for kind in ("RandomForest", "LogReg"):
            for k in K_GRID:
                r = evaluate(build(kind, k), S, seed)
                info = {"classes": float("nan"), "min_class": float("nan"), "ncp": 0.0}
                if k > 1:
                    pipe = build(kind, k)(seed).fit(X_in, y_in)
                    mon = pipe.named_steps["anon"]
                    imp = pipe.named_steps["impute"].transform(X_in)
                    info = {"classes": mon.n_classes_, "min_class": mon.min_class_size_, "ncp": mon.ncp(imp)}
                rows.append({"seed": seed, "model": kind, "k": k, **info, **r})
                log(f"  {kind:<12} k={k:<3} classes={info['classes']} min={info['min_class']} ncp={info['ncp']:.3f} "
                    f"acc={r['test_acc']:.3f} shadowAUC={r['shadow_auc']:.3f} lossAUC={r['loss_auc']:.3f}")
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    summ = summarise(runs, ["model", "k"], METRICS + ["ncp", "classes", "min_class"])
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    log("SUMMARY")
    for _, r in summ.iterrows():
        log(f"  {r['model']:<12} k={r['k']:<3} ncp={r['ncp_mean']:.3f} acc={r['test_acc_mean']:.3f}±{r['test_acc_ci95']:.3f} "
            f"shadowAUC={r['shadow_auc_mean']:.3f}±{r['shadow_auc_ci95']:.3f} min_class={r['min_class_mean']}")
    log("Done.")


if __name__ == "__main__":
    main()
