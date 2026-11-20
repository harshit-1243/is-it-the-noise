"""IA-1 preliminary experiments: membership inference vs. differential privacy on Pima diabetes.

Usage (from the project root):
    python src/run_experiments.py --data data/pima_diabetes.csv --out results --seeds 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attacks import loss_threshold_attack, shadow_model_attack          # noqa: E402
from data_prep import audit, clean, load_dataset, make_splits           # noqa: E402
from models import DP_MLP, calibrate_noise, compute_epsilon, make_model  # noqa: E402

DELTA = 1e-5
EPS_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]


class Logger:
    def __init__(self, path):
        self.f = open(path, "w")
        self.t0 = time.time()

    def __call__(self, msg=""):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def utility(model, X_in, y_in, X_out, y_out) -> dict:
    p = model.predict_proba(X_out)[:, 1]
    train_acc = model.score(X_in, y_in)
    test_acc = model.score(X_out, y_out)
    return {
        "train_acc": train_acc, "test_acc": test_acc, "gen_gap": train_acc - test_acc,
        "test_f1": f1_score(y_out, (p >= 0.5).astype(int)), "test_rocauc": roc_auc_score(y_out, p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pima_diabetes.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--shadows", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = Logger(os.path.join(args.out, "run_log.txt"))

    # ---------------- Stage 3: dataset audit & preparation ---------------- #
    raw = load_dataset(args.data)
    rep = audit(raw)
    json.dump(rep, open(os.path.join(args.out, "data_audit.json"), "w"), indent=2)
    log(f"Loaded {args.data}: {rep['rows']} rows x {rep['columns']} cols, "
        f"duplicates={rep['duplicate_rows']}, nulls={rep['null_values']}, classes={rep['class_counts']}")
    log(f"Zero-coded missing values: {rep['zero_coded_missing']}")
    df = clean(raw)
    log("Recoded impossible zeros -> NaN; median imputation + standardisation fitted inside each pipeline")

    # ---------------- DP-SGD calibration (same n for target & shadows) ---- #
    n_train = len(make_splits(df, 0)["target_in"][1])
    q = DP_MLP["batch_size"] / n_train
    steps = int(np.ceil(DP_MLP["epochs"] / q))
    dp_cfgs = [("DP-MLP (clip only)", float("inf"), 0.0)]
    for eps in EPS_GRID:
        sigma = calibrate_noise(eps, q, steps, DELTA)
        dp_cfgs.append((f"DP-MLP eps={eps:g}", compute_epsilon(q, sigma, steps, DELTA), sigma))
    log(f"DP-SGD: n={n_train}, sample rate q={q:.4f}, steps={steps}, clip C={DP_MLP['clip_norm']}, delta={DELTA}")
    for name, eps, sigma in dp_cfgs:
        log(f"  {name:<22} noise multiplier sigma={sigma:.3f}  -> achieved eps={eps:.3f}")

    configs = [("LogReg", "LogReg", {}, None), ("RandomForest", "RandomForest", {}, None), ("MLP", "MLP", {}, None)]
    configs += [(name, "DP-MLP", {"noise_multiplier": sigma}, eps) for name, eps, sigma in dp_cfgs]

    rows, seed0_scores = [], {}
    for seed in range(args.seeds):
        S = make_splits(df, seed)
        X_sh, y_sh = S["shadow"]; X_in, y_in = S["target_in"]; X_out, y_out = S["target_out"]
        log(f"--- seed {seed}: shadow={len(y_sh)}, target_in(members)={len(y_in)}, target_out(non-members/test)={len(y_out)}")
        for label, kind, kw, eps in configs:
            target = make_model(kind, seed, **kw).fit(X_in, y_in)
            u = utility(target, X_in, y_in, X_out, y_out)
            la = loss_threshold_attack(target, X_in, y_in, X_out, y_out)
            sa = shadow_model_attack(lambda s, k=kind, kw=kw: make_model(k, s, **kw), X_sh, y_sh,
                                     target, X_in, y_in, X_out, y_out, n_shadows=args.shadows, seed=seed)
            rows.append({
                "seed": seed, "model": label, "epsilon": eps if eps is not None else np.nan, **u,
                "loss_attack_auc": la["attack_auc"], "loss_attack_adv": la["max_advantage"],
                "loss_attack_tpr@5fpr": la["tpr_at_5fpr"],
                "shadow_attack_auc": sa["attack_auc"], "shadow_attack_acc@0.5": sa["attack_acc@0.5"],
                "shadow_attack_adv@0.5": sa["adv@0.5"],
            })
            if seed == 0:
                seed0_scores[label] = {"loss_in": la["scores_in"], "loss_out": la["scores_out"],
                                       "shadow_in": sa["scores_in"], "shadow_out": sa["scores_out"]}
            r = rows[-1]
            log(f"  {label:<22} train={r['train_acc']:.3f} test={r['test_acc']:.3f} gap={r['gen_gap']:+.3f} "
                f"F1={r['test_f1']:.3f} | loss-MIA AUC={r['loss_attack_auc']:.3f} adv={r['loss_attack_adv']:.3f} "
                f"| shadow-MIA AUC={r['shadow_attack_auc']:.3f} acc={r['shadow_attack_acc@0.5']:.3f}")

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(args.out, "all_runs.csv"), index=False)
    order = [c[0] for c in configs]
    metrics = ["train_acc", "test_acc", "gen_gap", "test_f1", "test_rocauc", "loss_attack_auc",
               "loss_attack_adv", "loss_attack_tpr@5fpr", "shadow_attack_auc", "shadow_attack_acc@0.5",
               "shadow_attack_adv@0.5"]
    g = runs.groupby("model", sort=False)
    summary = g[metrics].agg(["mean", "std"]).loc[order]
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "epsilon", g["epsilon"].first().loc[order])
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    np.savez(os.path.join(args.out, "scores_seed0.npz"),
             **{f"{k}|{kk}": vv for k, v in seed0_scores.items() for kk, vv in v.items()})

    log("")
    log(f"SUMMARY (mean +/- std over {args.seeds} seeds)")
    log(f"{'model':<22}{'eps':>7}{'test acc':>16}{'gen gap':>16}{'loss-MIA AUC':>16}{'shadow-MIA AUC':>16}")
    for m in order:
        s = summary.loc[m]
        e = "-" if pd.isna(s["epsilon"]) else ("inf" if np.isinf(s["epsilon"]) else f"{s['epsilon']:.2f}")
        log(f"{m:<22}{e:>7}"
            + "".join(f"{s[k + '_mean']:>9.3f}±{s[k + '_std']:.3f}" for k in
                      ["test_acc", "gen_gap", "loss_attack_auc", "shadow_attack_auc"]))
    log("Done.")


if __name__ == "__main__":
    main()
