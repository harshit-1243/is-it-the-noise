"""Re-runs the user's LiRA experiment (same code, same seed=1, so the same 64 masks and models) for the
configurations the paper compares, keeps the pooled per-decision scores, and adds record-level
(cluster) bootstrap 95% CIs plus a paired bootstrap for differences. Usage: python lira_bootstrap.py PROJECT OUTDIR"""
import sys, os, json
import numpy as np
proj, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, proj + '/src')
from data_prep import FEATURES, TARGET, clean, load_dataset
from lira import lira_metrics, lira_scores, train_population
from models import make_model
from sklearn.metrics import roc_curve
os.makedirs(out, exist_ok=True)
acc = json.load(open(proj + '/results/exp_lira/accounting.json'))
sig = {float(k): v for k, v in acc['sigma'].items()}
df = clean(load_dataset(proj + '/data/pima_diabetes.csv'))
X, y = df[FEATURES].reset_index(drop=True), df[TARGET].to_numpy()
cfgs = [("RandomForest", "RandomForest", {}), ("MLP", "MLP", {}), ("MLP-matched", "MLP-matched", {}),
        ("DP-MLP clip-only", "DP-MLP", {"noise_multiplier": 0.0}),
        ("DP-MLP eps=8", "DP-MLP", {"noise_multiplier": sig[8.0]}),
        ("DP-MLP eps=4", "DP-MLP", {"noise_multiplier": sig[4.0]}),
        ("DP-MLP eps=1", "DP-MLP", {"noise_multiplier": sig[1.0]}),
        ("LogReg", "LogReg", {}), ("DP-LR eps=4", "DP-LR", {"noise_multiplier": sig[4.0]})]
K, N = 64, len(y)
S = {}
for label, kind, kw in cfgs:
    masks, phi = train_population(lambda s, k=kind, kw=kw: make_model(k, s, **kw), X, y, K, seed=1)
    sc, lb = lira_scores(masks, phi)
    S[label] = (sc, lb)
    m = lira_metrics(sc, lb)
    print(label, {k: round(v, 4) for k, v in m.items()}, flush=True)
np.savez_compressed(os.path.join(out, 'lira_scores.npz'), **{k.replace(' ', '_'): v[0] for k, v in S.items()},
                    labels=S['RandomForest'][1])
# record index of each pooled decision: lira_scores concatenates K blocks of N records
rec = np.tile(np.arange(N), K)
def metr(sc, lb):
    fpr, tpr, _ = roc_curve(lb, sc)
    ok1, ok01 = fpr <= 0.01, fpr <= 0.001
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(lb, sc), tpr[ok1].max(), tpr[ok01].max()
rng = np.random.default_rng(2026)
B = 1000
boot = {k: [] for k in S}
for b in range(B):
    pick = rng.integers(0, N, N)                      # resample records with replacement
    idx = (np.arange(K)[:, None] * N + pick[None, :]).ravel()
    for k, (sc, lb) in S.items():
        boot[k].append(metr(sc[idx], lb[idx]))
res = {}
for k, (sc, lb) in S.items():
    a = np.array(boot[k]); pt = metr(sc, lb)
    res[k] = {"auc": pt[0], "tpr1": pt[1], "tpr01": pt[2],
              "auc_ci": list(np.percentile(a[:, 0], [2.5, 97.5])),
              "tpr1_ci": list(np.percentile(a[:, 1], [2.5, 97.5])),
              "tpr01_ci": list(np.percentile(a[:, 2], [2.5, 97.5]))}
pairs = [("MLP", "MLP-matched"), ("MLP-matched", "DP-MLP clip-only"), ("DP-MLP clip-only", "DP-MLP eps=4"),
         ("DP-MLP clip-only", "DP-MLP eps=8"), ("DP-MLP clip-only", "DP-MLP eps=1"), ("DP-MLP eps=4", "DP-MLP eps=1"),
         ("RandomForest", "MLP")]
diff = {}
for a_, b_ in pairs:
    A, Bb = np.array(boot[a_]), np.array(boot[b_])
    d1, da = A[:, 1] - Bb[:, 1], A[:, 0] - Bb[:, 0]
    diff[f"{a_} - {b_}"] = {"tpr1_diff": res[a_]["tpr1"] - res[b_]["tpr1"], "tpr1_diff_ci": list(np.percentile(d1, [2.5, 97.5])),
                            "tpr1_frac_le0": float(np.mean(d1 <= 0)),
                            "auc_diff": res[a_]["auc"] - res[b_]["auc"], "auc_diff_ci": list(np.percentile(da, [2.5, 97.5])),
                            "auc_frac_le0": float(np.mean(da <= 0))}
json.dump({"B": B, "per_config": res, "paired": diff}, open(os.path.join(out, 'lira_bootstrap.json'), 'w'), indent=1)
print(json.dumps(diff, indent=1))
