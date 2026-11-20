"""EXP-C (Module 3): local DP (patients perturb their own records) vs central DP.

label-LDP : randomised response on the diagnosis only (features shared as-is)
full-LDP  : randomised response on the diagnosis + Laplace on all 8 features (eps split 9 ways)
The learner is logistic regression trained on the noisy reports; predictions use clean inputs.
Central-DP comparison rows (DP-LR, DP-MLP) come from EXP-A at the same epsilon.

Usage: python src/exp_ldp.py --seeds 10 --out results/exp_ldp
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
from ldp import LDPTrained                                # noqa: E402
from models import make_model                             # noqa: E402

EPS_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pima_diabetes.csv")
    ap.add_argument("--out", default="results/exp_ldp")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    df = clean(load_dataset(a.data))
    rows = []
    for seed in range(a.seeds):
        S = make_splits(df, seed)
        log(f"--- seed {seed}")
        for mode in ("label", "full"):
            for eps in EPS_GRID:
                fn = (lambda s, e=eps, md=mode: LDPTrained(make_model("LogReg", s), eps=e, mode=md, seed=s))
                r = evaluate(fn, S, seed)
                rows.append({"seed": seed, "model": f"LDP-{mode}", "mode": mode, "epsilon": eps, **r})
                log(f"  LDP-{mode:<5} eps={eps:<4g} acc={r['test_acc']:.3f} f1={r['f1']:.3f} "
                    f"rocAUC={r['roc_auc']:.3f} shadowAUC={r['shadow_auc']:.3f}")
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    summ = summarise(runs, ["model", "mode", "epsilon"], METRICS)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    log("SUMMARY")
    for _, r in summ.iterrows():
        log(f"  {r['model']:<10} eps={r['epsilon']:<4g} acc={r['test_acc_mean']:.3f}±{r['test_acc_ci95']:.3f} "
            f"f1={r['f1_mean']:.3f} rocAUC={r['roc_auc_mean']:.3f} shadowAUC={r['shadow_auc_mean']:.3f}")
    log("Done.")


if __name__ == "__main__":
    main()
