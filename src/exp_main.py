"""EXP-A (Modules 1-2): baselines vs DP-SGD on Pima, 10 seeds, 95% CIs and paired tests.

New in IA-2 vs IA-1: 10 seeds; precision/recall; matched-recipe and L2 baselines that
separate *training length*, *clipping* and *noise*; DP logistic regression at every epsilon.

Usage: python src/exp_main.py --seeds 10 --out results/exp_main
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import audit, clean, load_dataset, make_splits   # noqa: E402
from evaluate import Logger, evaluate, paired_test, summarise    # noqa: E402
from models import DP_LR, DP_MLP, calibrate_noise, compute_epsilon, make_model  # noqa: E402

DELTA = 1e-5
EPS_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
METRICS = ["train_acc", "test_acc", "gen_gap", "precision", "recall", "f1", "roc_auc",
           "loss_auc", "loss_adv", "loss_tpr5", "shadow_auc", "shadow_acc", "shadow_adv"]


def configs(n_train: int):
    q = DP_MLP["batch_size"] / n_train
    steps = int(np.ceil(DP_MLP["epochs"] / q))
    out = [("LogReg", "LogReg", {}, np.nan, "non-private"),
           ("RandomForest", "RandomForest", {}, np.nan, "non-private"),
           ("MLP", "MLP", {}, np.nan, "non-private"),
           ("MLP-L2", "MLP-L2", {}, np.nan, "non-private, regularised"),
           ("MLP-matched", "MLP-matched", {}, np.nan, "non-private, DP recipe w/o clip+noise"),
           ("DP-MLP clip-only", "DP-MLP", {"noise_multiplier": 0.0}, np.inf, "clipping, no noise")]
    sig = {}
    for e in EPS_GRID:
        s = calibrate_noise(e, q, steps, DELTA)
        sig[e] = s
        out.append((f"DP-MLP eps={e:g}", "DP-MLP", {"noise_multiplier": s}, e, "DP-SGD"))
    for e in EPS_GRID:   # DP-LR uses the same q/steps (same batch size and epochs), so the same sigma
        out.append((f"DP-LR eps={e:g}", "DP-LR", {"noise_multiplier": sig[e]}, e, "DP-SGD"))
    return out, {"q": q, "steps": steps, "delta": DELTA,
                 "sigma": {str(k): v for k, v in sig.items()},
                 "achieved_eps": {str(k): compute_epsilon(q, v, steps, DELTA) for k, v in sig.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pima_diabetes.csv")
    ap.add_argument("--out", default="results/exp_main")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--shadows", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    raw = load_dataset(a.data)
    log(f"Audit: {audit(raw)}")
    df = clean(raw)
    n_train = len(make_splits(df, 0)["target_in"][1])
    cfgs, acct = configs(n_train)
    json.dump(acct, open(os.path.join(a.out, "privacy_accounting.json"), "w"), indent=2)
    log(f"DP-SGD accounting (n={n_train}): q={acct['q']:.4f}, steps={acct['steps']}, delta={DELTA}")
    for e, s in acct["sigma"].items():
        log(f"  eps={e:>4}  sigma={s:.3f}  achieved eps={acct['achieved_eps'][e]:.3f}")

    rows = []
    for seed in range(a.seeds):
        S = make_splits(df, seed)
        log(f"--- seed {seed}")
        for label, kind, kw, eps, family in cfgs:
            r = evaluate(lambda s, k=kind, kw=kw: make_model(k, s, **kw), S, seed, a.shadows)
            rows.append({"seed": seed, "model": label, "epsilon": eps, "family": family, **r})
            log(f"  {label:<18} acc={r['test_acc']:.3f} f1={r['f1']:.3f} gap={r['gen_gap']:+.3f} "
                f"lossAUC={r['loss_auc']:.3f} shadowAUC={r['shadow_auc']:.3f}")
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    summ = summarise(runs, ["model", "epsilon", "family"], METRICS)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    pairs = [("MLP", "MLP-matched", "shadow_auc"), ("MLP-matched", "DP-MLP clip-only", "shadow_auc"),
             ("DP-MLP clip-only", "DP-MLP eps=4", "shadow_auc"), ("MLP", "DP-MLP eps=4", "shadow_auc"),
             ("MLP-L2", "DP-MLP eps=4", "shadow_auc"), ("MLP-L2", "DP-MLP eps=4", "test_acc"),
             ("MLP", "DP-MLP eps=4", "test_acc"), ("DP-MLP eps=1", "DP-LR eps=1", "test_acc"),
             ("DP-MLP eps=4", "DP-LR eps=4", "test_acc"), ("LogReg", "RandomForest", "shadow_auc"),
             ("LogReg", "DP-LR eps=4", "test_acc"), ("MLP", "MLP-matched", "loss_auc"),
             ("MLP-matched", "DP-MLP clip-only", "loss_auc"), ("DP-MLP clip-only", "DP-MLP eps=4", "loss_auc")]
    tests = pd.DataFrame([paired_test(runs, x, y, m) for x, y, m in pairs])
    tests.to_csv(os.path.join(a.out, "paired_tests.csv"), index=False)
    log("")
    log("SUMMARY (mean ± 95% CI over seeds)")
    for _, r in summ.iterrows():
        log(f"  {r['model']:<18} acc={r['test_acc_mean']:.3f}±{r['test_acc_ci95']:.3f} "
            f"f1={r['f1_mean']:.3f} shadowAUC={r['shadow_auc_mean']:.3f}±{r['shadow_auc_ci95']:.3f} "
            f"lossAUC={r['loss_auc_mean']:.3f}")
    log("PAIRED TESTS (Wilcoxon signed-rank)")
    for _, t in tests.iterrows():
        log(f"  {t['a']} vs {t['b']} [{t['metric']}]: {t['mean_a']:.3f} vs {t['mean_b']:.3f}, p={t['p_wilcoxon']:.4f}")
    log("Done.")


if __name__ == "__main__":
    main()
