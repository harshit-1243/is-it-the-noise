"""Writes every results table of the paper as a LaTeX fragment, straight from the result files,
plus derived.json with the few derived quantities quoted in the text (differences, shares).

Usage: python make_paper_tables.py PROJECT_DIR LIRA_BOOT_JSON EXTRA_JSON OUT_DIR
"""
import json
import os
import sys

import numpy as np
import pandas as pd

proj, boot_path, extra_path, out = sys.argv[1:5]
R = os.path.join(proj, "results")
os.makedirs(out, exist_ok=True)
B = json.load(open(boot_path))
boot, pairs = B["per_config"], B["paired"]
extra = json.load(open(extra_path))
m = pd.read_csv(f"{R}/exp_main/summary.csv").set_index("model")
t = pd.read_csv(f"{R}/exp_main/paired_tests.csv")
lira = pd.read_csv(f"{R}/exp_lira/summary.csv").set_index("model")
ld = pd.read_csv(f"{R}/exp_ldp/summary.csv")
kn = pd.read_csv(f"{R}/exp_kanon/summary.csv")
kr = pd.read_csv(f"{R}/exp_kanon/runs.csv")
cp = pd.read_csv(f"{R}/exp_compressive/summary.csv").set_index("config")
sc = pd.read_csv(f"{R}/exp_scaling/summary.csv")
se = pd.read_csv(f"{R}/exp_sensitivity/summary.csv")
acct = json.load(open(f"{R}/exp_main/privacy_accounting.json"))


def f3(x):
    return f"{x:.3f}"


def pm(row, k, d=3):
    return f"{row[k + '_mean']:.{d}f}$\\pm${row[k + '_ci95']:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def P(a, b, met):
    r = t[(t.a == a) & (t.b == b) & (t.metric == met)].iloc[0]
    return r


def pfmt(p):
    return f"{p:.3f}"


def write(name, s):
    open(os.path.join(out, name + ".tex"), "w").write(s)


NAMES = {"LogReg": "Logistic regression", "RandomForest": "Random Forest", "MLP": "MLP, 1000 epochs",
         "MLP-L2": "MLP + L2 ($\\lambda{=}10^{-2}$)", "MLP-matched": "MLP, 30 epochs (matched)",
         "DP-MLP clip-only": "MLP, 30 ep.\\ + clipping ($\\sigma{=}0$)"}

# ---------------------------------------------------------------- LiRA table
rows = [("RandomForest", "Random Forest"), ("MLP", "MLP, 1000 epochs"), ("MLP-matched", "MLP, 30 epochs (matched)"),
        ("DP-MLP clip-only", "+ clipping, no noise"), ("DP-MLP eps=8", "DP-SGD MLP, $\\varepsilon{=}8$"),
        ("DP-MLP eps=4", "DP-SGD MLP, $\\varepsilon{=}4$"), ("DP-MLP eps=1", "DP-SGD MLP, $\\varepsilon{=}1$"),
        ("DP-LR eps=4", "DP-SGD LR, $\\varepsilon{=}4$"), ("LogReg", "Logistic regression")]
L = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
     "Target model (384 records) & LiRA AUC & TPR@1\\% (\\%) & TPR@0.1\\% (\\%) \\\\", "\\midrule"]
for key, lab in rows:
    v = boot[key]
    L.append(f"{lab} & {v['auc']:.3f} {{\\scriptsize[{v['auc_ci'][0]:.3f}, {v['auc_ci'][1]:.3f}]}} & "
             f"{pct(v['tpr1'])} {{\\scriptsize[{pct(v['tpr1_ci'][0])}, {pct(v['tpr1_ci'][1])}]}} & "
             f"{pct(v['tpr01'], 2)} {{\\scriptsize[{pct(v['tpr01_ci'][0], 2)}, {pct(v['tpr01_ci'][1], 2)}]}} \\\\")
    assert abs(v["auc"] - lira.loc[key, "lira_auc"]) < 1e-9 and abs(v["tpr1"] - lira.loc[key, "lira_tpr_at_1pct"]) < 1e-9
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_lira", "\n".join(L) + "\n")

# ---------------------------------------------------------------- decomposition table
dec = [("MLP", "MLP-matched", "Shorter training (1000$\\to$30 epochs)"),
       ("MLP-matched", "DP-MLP clip-only", "Per-record clipping ($C{=}1$)"),
       ("DP-MLP clip-only", "DP-MLP eps=4", "Gaussian noise ($\\varepsilon{=}4$)")]
L = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
     "Step added & Shadow AUC & Loss AUC & LiRA AUC \\\\",
     " & {\\scriptsize $\\Delta$ ($p$, Wilcoxon)} & {\\scriptsize $\\Delta$ ($p$)} & {\\scriptsize $\\Delta$ [95\\% CI]} \\\\", "\\midrule"]
derived = {}
for a, b, lab in dec:
    s = P(a, b, "shadow_auc"); lo = P(a, b, "loss_auc"); pr = pairs[f"{a} - {b}"]
    L.append(f"{lab} & $-${s.mean_diff:.3f} ({pfmt(s.p_wilcoxon)}) & $-${lo.mean_diff:.3f} ({pfmt(lo.p_wilcoxon)}) & "
             f"$-${pr['auc_diff']:.3f} {{\\scriptsize[{pr['auc_diff_ci'][0]:.3f}, {pr['auc_diff_ci'][1]:.3f}]}} \\\\")
tot_sh = m.loc["MLP", "shadow_auc_mean"] - m.loc["DP-MLP eps=4", "shadow_auc_mean"]
tot_li = boot["MLP"]["auc"] - boot["DP-MLP eps=4"]["auc"]
L.append("\\midrule")
L.append(f"Total (MLP $\\to$ DP-SGD, $\\varepsilon{{=}}4$) & $-${tot_sh:.3f} ({pfmt(P('MLP', 'DP-MLP eps=4', 'shadow_auc').p_wilcoxon)}) & "
         f"$-${m.loc['MLP', 'loss_auc_mean'] - m.loc['DP-MLP eps=4', 'loss_auc_mean']:.3f} & $-${tot_li:.3f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_decomp", "\n".join(L) + "\n")
for a, b, key in [("MLP", "MLP-matched", "length"), ("MLP-matched", "DP-MLP clip-only", "clip"), ("DP-MLP clip-only", "DP-MLP eps=4", "noise")]:
    derived[f"share_shadow_{key}"] = P(a, b, "shadow_auc").mean_diff / tot_sh
    derived[f"share_lira_{key}"] = pairs[f"{a} - {b}"]["auc_diff"] / tot_li
derived["total_drop_shadow"] = tot_sh; derived["total_drop_lira"] = tot_li

# ---------------------------------------------------------------- main grid (Pima, central DP and baselines)
grid = [("LogReg", "Logistic regression", "--"), ("RandomForest", "Random Forest", "--"), ("MLP", "MLP, 1000 epochs", "--"),
        ("MLP-L2", "MLP + L2", "--"), ("MLP-matched", "MLP, 30 ep.\\ (matched)", "--"), ("DP-MLP clip-only", "+ clipping, $\\sigma{=}0$", "$\\infty$")]
grid += [(f"DP-MLP eps={e:g}", "DP-SGD MLP", f"{e:g}") for e in (0.5, 1, 2, 4, 8)]
grid += [(f"DP-LR eps={e:g}", "DP-SGD LR", f"{e:g}") for e in (0.5, 1, 2, 4, 8)]
L = ["\\begin{tabular}{@{}llccccccc@{}}", "\\toprule",
     "Model & $\\varepsilon$ & $\\sigma$ & Accuracy & Recall & F1 & ROC-AUC & Gap & Shadow AUC \\\\", "\\midrule"]
for i, (key, lab, e) in enumerate(grid):
    r = m.loc[key]
    sig = "--" if e in ("--",) else ("0" if e == "$\\infty$" else f"{acct['sigma'][str(float(e))]:.2f}")
    if i in (6, 11):
        L.append("\\midrule")
    L.append(f"{lab} & {e} & {sig} & {pm(r, 'test_acc')} & {f3(r.recall_mean)} & {f3(r.f1_mean)} & {f3(r.roc_auc_mean)} & "
             f"{f3(r.gen_gap_mean)} & {pm(r, 'shadow_auc')} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_grid", "\n".join(L) + "\n")

# ---------------------------------------------------------------- BRFSS scaling
ns = sorted(sc["n"].unique())
def S(model, e, n):
    q = sc[(sc.model == model) & (sc.n == n)]
    if e is not None:
        q = q[q.epsilon == e]
    return q.iloc[0]["roc_auc_mean"]
L = ["\\begin{tabular}{@{}rccccc@{}}", "\\toprule",
     "$n$ & Non-private & No noise & $\\varepsilon{=}4$ & $\\varepsilon{=}1$ & $\\varepsilon{=}0.5$ \\\\", "\\midrule"]
noise_cost = {}
for n in ns:
    np_, cl = S("LogReg (non-private)", None, n), S("DP-LR clip-only (no noise)", None, n)
    cells = [f"{np_:.3f}", f"{cl:.3f}"]
    for e in (4.0, 1.0, 0.5):
        v = S("DP-LR", e, n)
        dd = v - cl
        cells.append(f"{v:.3f} {{\\scriptsize({'0.000' if abs(dd) < 5e-4 else f'{dd:+.3f}'})}}")
        noise_cost[f"{n}_{e:g}"] = cl - v
    L.append(f"{n:,} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_scaling", "\n".join(L).replace("+0.", "+0.").replace("-0.", "$-$0.") + "\n")
derived["noise_cost"] = noise_cost
derived["optimiser_gap"] = {str(n): S("LogReg (non-private)", None, n) - S("DP-LR clip-only (no noise)", None, n) for n in ns}

# ---------------------------------------------------------------- sensitivity
L = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule", "Setting ($\\varepsilon{=}4$) & $\\sigma$ & Accuracy & F1 & Shadow AUC \\\\", "\\midrule"]
for _, r in se.iterrows():
    lab = f"$C{{=}}{r['clip']:g}$, 30 epochs" if r["vary"] == "clip" else f"$C{{=}}1$, {int(r['epochs'])} epochs"
    if r["vary"] == "clip" and r["clip"] == 1.0:
        lab += " (used)"
    L.append(f"{lab} & {r['sigma']:.3f} & {pm(r, 'test_acc')} & {f3(r.f1_mean)} & {f3(r.shadow_auc_mean)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_sens", "\n".join(L) + "\n")

# ---------------------------------------------------------------- compressive releases
cfg = [("raw (8-D)", "Raw features (8-D)"), ("PCA m=4", "PCA, $m{=}4$"), ("PCA m=1", "PCA, $m{=}1$"), ("DCA m=1", "DCA, $m{=}1$"),
       ("PDCA m=1 b=1", "PDCA, $\\beta{=}1$"), ("PDCA m=1 b=3", "PDCA, $\\beta{=}3$"), ("PDCA m=1 b=10", "PDCA, $\\beta{=}10$")]
L = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
     "Release $Z$ & Acc.\\ (LR on $Z$) & ROC-AUC & Age AUC & $R^2$ & RF shadow AUC \\\\", "\\midrule"]
for key, lab in cfg:
    r = cp.loc[key]
    L.append(f"{lab} & {pm(r, 'test_acc')} & {f3(r.roc_auc_mean)} & {pm(r, 'attr_auc')} & {r.recon_r2_mean:.2f} & {pm(r, 'rf_shadow_auc')} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_cp", "\n".join(L) + "\n")

# ---------------------------------------------------------------- same-adversary comparison
def kget(model, k, col):
    return kn[(kn.model == model) & (kn.k == k)].iloc[0][col]
def lget(mode, e, col):
    return ld[(ld["mode"] == mode) & (ld.epsilon == e)].iloc[0][col]
rows = [
    ("None", "Logistic regression", m.loc["LogReg", "test_acc_mean"], m.loc["LogReg", "roc_auc_mean"], m.loc["LogReg", "shadow_auc_mean"], boot["LogReg"]["tpr1"], "none"),
    ("None", "Random Forest", m.loc["RandomForest", "test_acc_mean"], m.loc["RandomForest", "roc_auc_mean"], m.loc["RandomForest", "shadow_auc_mean"], boot["RandomForest"]["tpr1"], "none"),
    ("Central DP", "DP-SGD LR, $\\varepsilon{=}4$", m.loc["DP-LR eps=4", "test_acc_mean"], m.loc["DP-LR eps=4", "roc_auc_mean"], m.loc["DP-LR eps=4", "shadow_auc_mean"], boot["DP-LR eps=4"]["tpr1"], "$(4,10^{-5})$-DP"),
    ("Central DP", "DP-SGD LR, $\\varepsilon{=}1$", m.loc["DP-LR eps=1", "test_acc_mean"], m.loc["DP-LR eps=1", "roc_auc_mean"], m.loc["DP-LR eps=1", "shadow_auc_mean"], None, "$(1,10^{-5})$-DP"),
    ("Central DP", "DP-SGD MLP, $\\varepsilon{=}4$", m.loc["DP-MLP eps=4", "test_acc_mean"], m.loc["DP-MLP eps=4", "roc_auc_mean"], m.loc["DP-MLP eps=4", "shadow_auc_mean"], boot["DP-MLP eps=4"]["tpr1"], "$(4,10^{-5})$-DP"),
    ("Local DP", "Label only (RR), $\\varepsilon{=}4$", lget("label", 4.0, "test_acc_mean"), lget("label", 4.0, "roc_auc_mean"), lget("label", 4.0, "shadow_auc_mean"), None, "4-LDP, label"),
    ("Local DP", "All attributes, $\\varepsilon{=}16$", lget("full", 16.0, "test_acc_mean"), lget("full", 16.0, "roc_auc_mean"), lget("full", 16.0, "shadow_auc_mean"), None, "16-LDP, record"),
    ("$k$-anonymity", "$k{=}5$ + Random Forest", kget("RandomForest", 5, "test_acc_mean"), kget("RandomForest", 5, "roc_auc_mean"), kget("RandomForest", 5, "shadow_auc_mean"), None, "syntactic"),
    ("$k$-anonymity", "$k{=}5$ + LR", kget("LogReg", 5, "test_acc_mean"), kget("LogReg", 5, "roc_auc_mean"), kget("LogReg", 5, "shadow_auc_mean"), None, "syntactic"),
    ("Compressive", "DCA, $m{=}1$ (RF on $Z$)$^\\dagger$", cp.loc["DCA m=1", "test_acc_mean"], cp.loc["DCA m=1", "roc_auc_mean"], cp.loc["DCA m=1", "rf_shadow_auc_mean"], None, "none (empirical)"),
    ("Compressive", "PDCA, $\\beta{=}10$ (RF on $Z$)$^\\dagger$", cp.loc["PDCA m=1 b=10", "test_acc_mean"], cp.loc["PDCA m=1 b=10", "roc_auc_mean"], cp.loc["PDCA m=1 b=10", "rf_shadow_auc_mean"], None, "none (empirical)"),
]
L = ["\\begin{tabular}{@{}llccccl@{}}", "\\toprule",
     "Mechanism & Setting & Acc. & ROC-AUC & Shadow AUC & LiRA TPR@1\\% & Formal guarantee \\\\", "\\midrule"]
prev = None
for mech, sett, acc, auc, sh, tp, g in rows:
    if prev is not None and mech != prev:
        L.append("\\addlinespace[2pt]")
    tps = "--" if tp is None else f"{pct(tp)}\\%"
    L.append(f"{mech if mech != prev else ''} & {sett} & {acc:.3f} & {auc:.3f} & {sh:.3f} & {tps} & {g} \\\\")
    prev = mech
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_compare", "\n".join(L) + "\n")

# ---------------------------------------------------------------- k-anonymity paired tests (re-derived from runs.csv)
from scipy import stats
kt = {}
for model in ("RandomForest", "LogReg"):
    base = kr[(kr.model == model) & (kr.k == 1)].set_index("seed")
    for kk in (5, 50):
        b = kr[(kr.model == model) & (kr.k == kk)].set_index("seed")
        for met in ("shadow_auc", "test_acc"):
            kt[f"{model}_k{kk}_{met}"] = float(stats.wilcoxon(base[met], b.loc[base.index, met]).pvalue)
derived["kanon_p"] = kt
derived["ldp_full_eps4_per_attr_eps"] = 4 / 9
derived["ldp_full_eps4_laplace_sd_in_ranges"] = float(np.sqrt(2) * 9 / 4)
derived["lira_noise_tpr1_diff_pp"] = 100 * pairs["DP-MLP clip-only - DP-MLP eps=4"]["tpr1_diff"]
derived["lira_noise_tpr1_diff_ci_pp"] = [100 * x for x in pairs["DP-MLP clip-only - DP-MLP eps=4"]["tpr1_diff_ci"]]
derived["extra"] = extra
json.dump(derived, open(os.path.join(out, "derived.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in derived.items() if k.startswith(("share", "total"))}, indent=1, default=float))
print("noise cost", {k: round(v, 3) for k, v in noise_cost.items()})
print("optimiser gap", {k: round(v, 3) for k, v in derived["optimiser_gap"].items()})
print("kanon p", {k: round(v, 3) for k, v in kt.items()})
