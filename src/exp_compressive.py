"""EXP-E (Module 5): compressive privacy. The hospital releases Z = projection(X) instead of X.

For each projection we measure, on the 192 members of each split:
  utility          - logistic regression on Z: test accuracy, F1, ROC-AUC (non-members)
  attribute leak   - attacker trains logistic regression on its own shadow data projected with the
                     (public) projection to infer the private attribute age >= 35; accuracy on members
  reconstruction   - attacker fits a linear map Z -> X on shadow data; R^2 on members (all features, and Age)
  membership leak  - Random Forest trained on Z, attacked with loss + shadow MIAs (shadows replay the recipe)

Usage: python src/exp_compressive.py --seeds 10 --out results/exp_compressive
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compressive import AGE_COL, Projection, private_labels   # noqa: E402
from data_prep import clean, load_dataset, make_splits         # noqa: E402
from evaluate import Logger, evaluate, summarise, utility       # noqa: E402
from models import make_model                                   # noqa: E402

CFGS = [("raw (8-D)", "none", 8, 0.0), ("PCA m=4", "pca", 4, 0.0), ("PCA m=3", "pca", 3, 0.0),
        ("PCA m=2", "pca", 2, 0.0), ("PCA m=1", "pca", 1, 0.0), ("DCA m=1", "dca", 1, 0.0),
        ("PDCA m=1 b=1", "pdca", 1, 1.0), ("PDCA m=1 b=3", "pdca", 1, 3.0), ("PDCA m=1 b=10", "pdca", 1, 10.0)]


def build(kind, method, m, beta):
    def fn(s):
        return make_model(kind, s, extra_steps=[("proj", Projection(method=method, m=m, beta=beta))])
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pima_diabetes.csv")
    ap.add_argument("--out", default="results/exp_compressive")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    df = clean(load_dataset(a.data))
    rows = []
    for seed in range(a.seeds):
        S = make_splits(df, seed)
        X_sh, y_sh = S["shadow"]; X_in, y_in = S["target_in"]; X_out, y_out = S["target_out"]
        log(f"--- seed {seed}")
        for label, method, m, beta in CFGS:
            # utility: logistic regression on the released projection
            lr_pipe = build("LogReg", method, m, beta)(seed).fit(X_in, y_in)
            u = utility(lr_pipe, X_in, y_in, X_out, y_out)
            imp, proj = lr_pipe.named_steps["impute"], lr_pipe.named_steps["proj"]
            Z_in = proj.transform(imp.transform(X_in))
            Z_sh = proj.transform(imp.transform(X_sh))
            # attribute inference (age >= 35) from Z
            a_sh, a_in = private_labels(imp.transform(X_sh)), private_labels(imp.transform(X_in))
            att = LogisticRegression(max_iter=2000).fit(Z_sh, a_sh)
            attr_acc = float(np.mean(att.predict(Z_in) == a_in))
            attr_auc = float(roc_auc_score(a_in, att.predict_proba(Z_in)[:, 1]))
            majority = float(max(a_in.mean(), 1 - a_in.mean()))
            # reconstruction of the standardised raw features from Z
            Xs_sh, Xs_in = proj._std(imp.transform(X_sh)), proj._std(imp.transform(X_in))
            rec = LinearRegression().fit(Z_sh, Xs_sh)
            R = rec.predict(Z_in)
            r2_all = float(r2_score(Xs_in, R, multioutput="uniform_average"))
            r2_age = float(r2_score(Xs_in[:, AGE_COL], R[:, AGE_COL]))
            # membership leakage of a high-capacity model trained on Z
            mia = evaluate(build("RandomForest", method, m, beta), S, seed)
            rows.append({"seed": seed, "config": label, "method": method, "m": m, "beta": beta,
                         "test_acc": u["test_acc"], "f1": u["f1"], "roc_auc": u["roc_auc"],
                         "attr_acc": attr_acc, "attr_auc": attr_auc, "attr_majority": majority,
                         "recon_r2": r2_all, "recon_r2_age": r2_age,
                         "rf_test_acc": mia["test_acc"], "rf_shadow_auc": mia["shadow_auc"],
                         "rf_loss_auc": mia["loss_auc"]})
            log(f"  {label:<14} acc={u['test_acc']:.3f} rocAUC={u['roc_auc']:.3f} | age-inference acc={attr_acc:.3f} "
                f"(majority {majority:.3f}) AUC={attr_auc:.3f} | recon R2={r2_all:.3f} (age {r2_age:.3f}) | "
                f"RF shadowAUC={mia['shadow_auc']:.3f}")
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    metrics = ["test_acc", "f1", "roc_auc", "attr_acc", "attr_auc", "attr_majority", "recon_r2", "recon_r2_age",
               "rf_test_acc", "rf_shadow_auc", "rf_loss_auc"]
    summ = summarise(runs, ["config", "method", "m", "beta"], metrics)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    log("SUMMARY")
    for _, r in summ.iterrows():
        log(f"  {r['config']:<14} acc={r['test_acc_mean']:.3f}±{r['test_acc_ci95']:.3f} "
            f"ageAUC={r['attr_auc_mean']:.3f} reconR2={r['recon_r2_mean']:.3f} RFshadowAUC={r['rf_shadow_auc_mean']:.3f}")
    log("Done.")


if __name__ == "__main__":
    main()
