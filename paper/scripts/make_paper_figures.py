"""Builds every data figure of the paper directly from the result files (no number is retyped).

Usage: python make_paper_figures.py PROJECT_DIR LIRA_BOOT_JSON OUT_DIR
  PROJECT_DIR     the IA-2 project folder (reads results/*)
  LIRA_BOOT_JSON  output of lira_bootstrap.py (record-level bootstrap CIs for LiRA)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

proj, boot_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
R = os.path.join(proj, "results")
os.makedirs(out, exist_ok=True)

for f in ["/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.4, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
# validated categorical slots (dataviz reference palette, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK2, MUTED = "#52514e", "#8a8984"
SEQ = {"250": "#86b6ef", "450": "#2a78d6", "650": "#104281"}
COL_W, TXT_W = 3.45, 7.0

main = pd.read_csv(f"{R}/exp_main/summary.csv").set_index("model")
boot = json.load(open(boot_path))["per_config"]


def save(fig, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2: LiRA
roc = np.load(f"{R}/exp_lira/roc_curves.npz")
grid = roc["fpr_grid"]
curves = [("Random Forest", "RandomForest", C[1], "-"), ("MLP, 1000 epochs", "MLP", C[2], "-"),
          ("MLP, 30 epochs (matched)", "MLP-matched", C[3], "--"), ("+ clipping (no noise)", "DP-MLP_clip-only", C[4], "-."),
          (r"+ noise, DP-SGD $\varepsilon{=}4$", "DP-MLP_eps=4", C[5], ":"), ("Logistic regression", "LogReg", C[0], "-")]
fig = plt.figure(figsize=(TXT_W, 2.35))
a = fig.add_axes([0.065, 0.17, 0.27, 0.72])
leg_ax = fig.add_axes([0.345, 0.17, 0.17, 0.72]); leg_ax.axis("off")
b = fig.add_axes([0.66, 0.17, 0.30, 0.72])
a.plot(grid, grid, color=MUTED, lw=0.8, ls=(0, (2, 2)), label="chance")
for lab, key, col, ls in curves:
    a.plot(grid, np.maximum(roc[key], 1e-4), color=col, ls=ls, lw=1.5, label=lab)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1e-4, 1); a.set_ylim(1e-4, 1)
a.set_xlabel("False-positive rate"); a.set_ylabel("True-positive rate")
a.axvline(0.01, color="#c3c2b7", lw=0.6); a.axvline(0.001, color="#c3c2b7", lw=0.6)
a.text(0.0105, 1.6e-4, "1%", fontsize=6.5, color=INK2); a.text(0.00105, 1.6e-4, "0.1%", fontsize=6.5, color=INK2)
h, l = a.get_legend_handles_labels()
leg_ax.legend(h, l, loc="center left", handlelength=2.4, borderaxespad=0.0, labelspacing=0.7)
a.set_title("(a) LiRA ROC, log axes", loc="left")
rows = [("Random Forest", "RandomForest"), ("MLP, 1000 ep.", "MLP"), ("MLP + L2", None), ("MLP, 30 ep. (matched)", "MLP-matched"),
        ("+ clipping (no noise)", "DP-MLP clip-only"), (r"DP-MLP $\varepsilon{=}8$", "DP-MLP eps=8"),
        (r"DP-MLP $\varepsilon{=}4$", "DP-MLP eps=4"), (r"DP-MLP $\varepsilon{=}1$", "DP-MLP eps=1"),
        (r"DP-LR $\varepsilon{=}4$", "DP-LR eps=4"), ("Logistic regression", "LogReg")]
rows = [r for r in rows if r[1] is not None]
y = np.arange(len(rows))[::-1]
for yi, (lab, key) in zip(y, rows):
    v = boot[key]
    lo, hi = v["tpr1_ci"]
    b.plot([lo, hi], [yi, yi], color=C[0], lw=1.6, solid_capstyle="butt")
    b.plot(v["tpr1"], yi, "o", color=C[0], ms=4.2, zorder=3)
    b.text(hi * 1.12, yi, f"{100 * v['tpr1']:.1f}%", va="center", fontsize=6.5, color=INK2)
b.axvline(0.01, color=MUTED, lw=0.8, ls=(0, (2, 2)))
b.text(0.0102, len(rows) - 0.35, "chance (1%)", fontsize=6.5, color=INK2, va="bottom")
b.set_yticks(y); b.set_yticklabels([r[0] for r in rows]); b.set_xscale("log"); b.set_xlim(0.006, 1.2)
b.set_ylim(-0.6, len(rows) - 0.1)
b.set_xlabel("TPR at 1% FPR (95% bootstrap CI)")
b.set_title("(b) Members exposed at 1% FPR", loc="left")
b.tick_params(axis="y", length=0)
save(fig, "fig_lira")

# ------------------------------------------------------------------ Fig. 3: decomposition
steps = [("MLP, 1000 epochs", "MLP"), ("30 epochs (matched)", "MLP-matched"),
         ("+ clipping", "DP-MLP clip-only"), (r"+ noise ($\varepsilon{=}4$)", "DP-MLP eps=4")]
fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.75), sharey=True, gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.12})
y = np.arange(len(steps))[::-1]
off = 0.13
for yi, (lab, key) in zip(y, steps):
    m, h = main.loc[key, "shadow_auc_mean"], main.loc[key, "shadow_auc_ci95"]
    a.plot([m - h, m + h], [yi + off] * 2, color=C[0], lw=1.2)
    a.plot(m, yi + off, "o", color=C[0], ms=4.2, mec="white", mew=0.7, zorder=3, label="Shadow (192 rec., 10 splits)" if yi == y[0] else None)
    v = boot[key]
    a.plot(v["auc_ci"], [yi - off] * 2, color=C[1], lw=1.2)
    a.plot(v["auc"], yi - off, "s", color=C[1], ms=4, mec="white", mew=0.7, zorder=3, label="LiRA (384 rec., bootstrap)" if yi == y[0] else None)
    m2, h2 = main.loc[key, "test_acc_mean"], main.loc[key, "test_acc_ci95"]
    b.plot([m2 - h2, m2 + h2], [yi] * 2, color=INK2, lw=1.2)
    b.plot(m2, yi, "o", color=INK2, ms=4.2, mec="white", mew=0.7, zorder=3)
a.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
a.set_yticks(y); a.set_yticklabels([s[0] for s in steps]); a.tick_params(axis="y", length=0)
a.set_xlabel("Attack AUC"); a.set_xlim(0.48, 0.78)
a.legend(loc="upper center", bbox_to_anchor=(0.62, 1.32), ncol=1, handletextpad=0.3)
b.set_xlabel("Test accuracy"); b.set_xlim(0.67, 0.79)
b.tick_params(axis="y", length=0)
save(fig, "fig_decomposition")

# ------------------------------------------------------------------ Fig. 4: central-DP frontier (accuracy)
eps = [0.5, 1, 2, 4, 8]
fig, a = plt.subplots(figsize=(COL_W, 1.85))
for fam, col, mk, lab in [("DP-LR", C[0], "o", "DP-SGD logistic regression"), ("DP-MLP", C[2], "s", "DP-SGD MLP")]:
    m = np.array([main.loc[f"{fam} eps={e:g}", "test_acc_mean"] for e in eps])
    h = np.array([main.loc[f"{fam} eps={e:g}", "test_acc_ci95"] for e in eps])
    a.fill_between(eps, m - h, m + h, color=col, alpha=0.15, lw=0)
    a.plot(eps, m, marker=mk, color=col, ms=4, mec="white", mew=0.7, label=lab)
a.axhline(main.loc["LogReg", "test_acc_mean"], color=C[0], lw=0.9, ls=(0, (3, 2)))
a.text(8.4, main.loc["LogReg", "test_acc_mean"] + 0.003, "non-private LR", fontsize=6.5, color=INK2, ha="right", va="bottom")
a.axhline(main.loc["MLP", "test_acc_mean"], color=C[2], lw=0.9, ls=(0, (3, 2)))
a.text(8.4, main.loc["MLP", "test_acc_mean"] - 0.004, "non-private MLP (1000 ep.)", fontsize=6.5, color=INK2, ha="right", va="top")
a.set_xscale("log", base=2); a.set_xticks(eps); a.set_xticklabels(["0.5", "1", "2", "4", "8"])
a.set_xlabel(r"Privacy budget $\varepsilon$ ($\delta=10^{-5}$)"); a.set_ylabel("Test accuracy")
a.set_ylim(0.55, 0.80); a.legend(loc="lower right")
a.grid(axis="y")
save(fig, "fig_frontier")

# ------------------------------------------------------------------ Fig. 5: BRFSS scaling
sc = pd.read_csv(f"{R}/exp_scaling/summary.csv")
ns = sorted(sc["n"].unique())
fig, a = plt.subplots(figsize=(COL_W, 2.1))
def ser(model, e=None):
    q = sc[sc.model == model]
    if e is not None:
        q = q[q.epsilon == e]
    q = q.set_index("n").loc[ns]
    return q["roc_auc_mean"].to_numpy(), q["roc_auc_ci95"].to_numpy()
spec = [("LogReg (non-private)", None, "#0b0b0b", (0, (3, 2)), "o", "non-private LR"),
        ("DP-LR clip-only (no noise)", None, MUTED, "-", "D", "DP optimiser, no noise"),
        ("DP-LR", 4.0, SEQ["650"], "-", "o", r"DP-LR $\varepsilon{=}4$"),
        ("DP-LR", 1.0, SEQ["450"], "-", "s", r"DP-LR $\varepsilon{=}1$"),
        ("DP-LR", 0.5, SEQ["250"], "-", "^", r"DP-LR $\varepsilon{=}0.5$")]
for k, (model, e, col, ls, mk, lab) in enumerate(spec):
    m, h = ser(model, e)
    x = np.array(ns) * (1 + 0.04 * (k - 2))
    a.errorbar(x, m, yerr=h, color=col, ls=ls, marker=mk, ms=3.6, mec="white", mew=0.6, lw=1.3, elinewidth=0.7, capsize=0, label=lab)
a.set_xscale("log"); a.set_xticks(ns); a.set_xticklabels(["200", "1k", "5k", "25k", "100k"])
a.set_xlabel("Training records $n$ (BRFSS 2015)"); a.set_ylabel("Test ROC-AUC (20k held out)")
a.set_ylim(0.30, 0.90); a.grid(axis="y"); a.legend(loc="lower right", ncol=1, labelspacing=0.35)
save(fig, "fig_scaling")

# ------------------------------------------------------------------ Fig. 6: defences acting elsewhere
ld = pd.read_csv(f"{R}/exp_ldp/summary.csv")
kn = pd.read_csv(f"{R}/exp_kanon/summary.csv")
cp = pd.read_csv(f"{R}/exp_compressive/summary.csv").set_index("config")
fig, (a, b, c) = plt.subplots(1, 3, figsize=(TXT_W, 2.05), gridspec_kw={"wspace": 0.42})
e_c = [0.5, 1, 2, 4, 8]
a.plot(e_c, [main.loc[f"DP-LR eps={e:g}", "roc_auc_mean"] for e in e_c], marker="o", color=C[0], ms=3.8, mec="white", mew=0.6, label="central DP-LR")
for mode, col, mk, lab in [("label", C[1], "s", "LDP, label only"), ("full", C[2], "^", "LDP, all attributes")]:
    q = ld[ld["mode"] == mode].sort_values("epsilon")
    a.plot(q.epsilon, q.roc_auc_mean, marker=mk, color=col, ms=3.8, mec="white", mew=0.6, label=lab)
a.axhline(main.loc["LogReg", "roc_auc_mean"], color="#0b0b0b", lw=0.8, ls=(0, (3, 2)))
a.text(0.5, main.loc["LogReg", "roc_auc_mean"] + 0.006, "non-private", fontsize=6.5, color=INK2)
a.set_xscale("log", base=2); a.set_xticks([0.5, 1, 2, 4, 8, 16]); a.set_xticklabels(["0.5", "1", "2", "4", "8", "16"])
a.set_xlabel(r"$\varepsilon$ (per patient for local DP)"); a.set_ylabel("Diabetes ROC-AUC")
a.set_ylim(0.45, 0.88); a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.52), handlelength=1.8, labelspacing=0.3); a.grid(axis="y")
a.set_title("(a) Local vs central DP", loc="left")
ks = [1, 2, 5, 10, 25, 50]
for model, col, mk, lab in [("RandomForest", C[1], "s", "Random Forest"), ("LogReg", C[0], "o", "Logistic regression")]:
    q = kn[kn.model == model].set_index("k").loc[ks]
    xs = np.arange(len(ks))
    b.fill_between(xs, q.shadow_auc_mean - q.shadow_auc_ci95, q.shadow_auc_mean + q.shadow_auc_ci95, color=col, alpha=0.15, lw=0)
    b.plot(xs, q.shadow_auc_mean, marker=mk, color=col, ms=3.8, mec="white", mew=0.6, label=lab)
b.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
b.set_xticks(np.arange(len(ks))); b.set_xticklabels(["none", "2", "5", "10", "25", "50"])
b.set_xlabel("$k$ (Mondrian on Age, Pregnancies, BMI)"); b.set_ylabel("Shadow-attack AUC")
b.set_ylim(0.45, 0.86); b.legend(loc="center right"); b.grid(axis="y")
b.set_title("(b) $k$-anonymised training data", loc="left")
pts = [("raw (8-D)", "raw"), ("PCA m=4", "PCA 4"), ("PCA m=3", "PCA 3"), ("PCA m=2", "PCA 2"), ("PCA m=1", "PCA 1"),
       ("DCA m=1", "DCA"), ("PDCA m=1 b=1", r"PDCA $\beta{=}1$"), ("PDCA m=1 b=3", r"$\beta{=}3$"), ("PDCA m=1 b=10", r"$\beta{=}10$")]
offs = {"raw": (-4, 4, "right"), "PCA 4": (4, -6, "left"), "PCA 3": (4, 2, "left"), "PCA 2": (4, -7, "left"),
        "PCA 1": (-4, -7, "right"), "DCA": (4, 2, "left"), r"PDCA $\beta{=}1$": (4, 3, "left"), r"$\beta{=}3$": (3, 4, "left"),
        r"$\beta{=}10$": (2, -8, "left")}
for key, lab in pts:
    r = cp.loc[key]
    col = C[0] if key.startswith(("DCA", "PDCA")) else (C[1] if key.startswith("PCA") else INK2)
    mk = "o" if key.startswith(("DCA", "PDCA")) else ("s" if key.startswith("PCA") else "D")
    c.plot(r.attr_auc_mean, r.roc_auc_mean, mk, color=col, ms=4.2, mec="white", mew=0.6)
    dx, dy, ha = offs[lab]
    c.annotate(lab, (r.attr_auc_mean, r.roc_auc_mean), xytext=(dx, dy), textcoords="offset points", fontsize=6.3, color=INK2, ha=ha)
c.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
c.set_xlabel(r"Age $\geq$35 inference AUC (attacker)"); c.set_ylabel("Diabetes ROC-AUC (released $Z$)")
c.set_xlim(0.46, 1.10); c.set_ylim(0.785, 0.835); c.grid(alpha=0.6)
c.set_title("(c) Compressive releases", loc="left")
save(fig, "fig_dataside")
print("figures written to", out)
