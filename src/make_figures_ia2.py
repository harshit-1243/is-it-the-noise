"""Figures for the IA-2 report (Figures 2-9), built only from results/ CSV files.

Usage: python src/make_figures_ia2.py --results results --out figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GREY, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white", "legend.frameon": False,
})


def title(fig, main, sub, top=0.82):
    fig.tight_layout(rect=[0, 0, 1, top])
    h = fig.get_figheight()
    fig.text(0.01, 0.985, main, fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 0.985 - 0.30 / h, sub, fontsize=9.2, color=INK2, ha="left", va="top")


def ref_line(ax, v, text, side="right", below=False, color=GREY):
    ax.axhline(v, color=color, lw=1, ls=(0, (4, 3)), zorder=1)
    x0, x1 = ax.get_xlim()
    ax.text(x1 if side == "right" else x0, v + (-0.004 if below else 0.004), text, color=INK2, fontsize=8,
            ha="right" if side == "right" else "left", va="top" if below else "bottom")


# --------------------------------------------------------------------------- #
def fig_decomposition(R, out):
    s = pd.read_csv(f"{R}/exp_main/summary.csv").set_index("model")
    order = ["MLP", "MLP-L2", "MLP-matched", "DP-MLP clip-only", "DP-MLP eps=8", "DP-MLP eps=4", "DP-MLP eps=1"]
    labels = ["MLP, 1000 epochs (no defence)", "+ L2 regularisation", "30-epoch DP recipe, no clip/noise",
              "+ per-record clipping (ε = ∞)", "+ noise: DP-SGD ε = 8", "DP-SGD ε = 4", "DP-SGD ε = 1"]
    d = s.loc[order]
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
    for ax, col, xl, ref in [(axes[0], "shadow_auc", "Shadow-model attack AUC (lower = more private)", 0.5),
                             (axes[1], "test_acc", "Test accuracy (higher = more useful)", None)]:
        colors = [GREY if not m.startswith("DP-MLP eps") else CAT[0] for m in order]
        ax.hlines(y, d[f"{col}_mean"] - d[f"{col}_ci95"], d[f"{col}_mean"] + d[f"{col}_ci95"], color=colors, lw=2)
        ax.scatter(d[f"{col}_mean"], y, color=colors, s=46, zorder=3, edgecolor="white", linewidth=1.2)
        for yi, v in zip(y, d[f"{col}_mean"]):
            ax.text(v, yi + 0.22, f"{v:.3f}", fontsize=8, color=INK, ha="center", va="bottom")
        ax.set_xlabel(xl)
        if ref:
            ax.axvline(ref, color=GREY, lw=1, ls=(0, (4, 3)))
            ax.text(ref + 0.003, y[-1] - 0.45, "random guess", fontsize=8, color=INK2, va="center")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    axes[0].set_ylim(-0.8, len(order) - 0.3)
    axes[0].set_xlim(0.47, 0.78)
    axes[1].set_xlim(0.55, 0.80)
    title(fig, "Where the leakage reduction comes from: training length, clipping, then noise",
          "Pima, 10 random splits; dot = mean, bar = 95% CI · grey = no formal guarantee, blue = DP-SGD with (ε, δ = 10⁻⁵)")
    fig.savefig(f"{out}/fig3_decomposition.png")
    plt.close(fig)


def fig_frontier(R, out):
    s = pd.read_csv(f"{R}/exp_main/summary.csv").set_index("model")
    eps = [0.5, 1, 2, 4, 8]
    x = np.arange(len(eps))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for ax, col, yl in [(axes[0], "test_acc", "Test accuracy"), (axes[1], "shadow_auc", "Shadow-model attack AUC")]:
        for i, (fam, name) in enumerate([("DP-MLP", "DP-SGD MLP"), ("DP-LR", "DP-SGD logistic regression")]):
            d = s.loc[[f"{fam} eps={e:g}" for e in eps]]
            ax.errorbar(x + (i - 0.5) * 0.12, d[f"{col}_mean"], yerr=d[f"{col}_ci95"], fmt="-o", color=CAT[i],
                        ms=5.5, lw=2, elinewidth=1.2, capsize=0, mec="white", mew=1, label=name, zorder=3)
        ax.set_xticks(x, [f"{e:g}" for e in eps])
        ax.set_xlabel("Privacy budget ε  (δ = 10⁻⁵)")
        ax.set_ylabel(yl)
        ax.set_xlim(-0.4, len(eps) - 0.4)
    axes[0].set_ylim(0.5, 0.82)
    axes[1].set_ylim(0.45, 0.85)
    ref_line(axes[0], s.loc["LogReg", "test_acc_mean"], f"non-private LogReg {s.loc['LogReg','test_acc_mean']:.3f}", side="left")
    ref_line(axes[0], s.loc["MLP", "test_acc_mean"], f"non-private MLP {s.loc['MLP','test_acc_mean']:.3f}", side="right", below=True)
    ref_line(axes[1], s.loc["MLP", "shadow_auc_mean"], f"non-private MLP {s.loc['MLP','shadow_auc_mean']:.3f}", side="right")
    ref_line(axes[1], s.loc["RandomForest", "shadow_auc_mean"], f"Random Forest {s.loc['RandomForest','shadow_auc_mean']:.3f}", side="right", below=True)
    ref_line(axes[1], 0.5, "random guess 0.5", side="right", below=True)
    axes[0].legend(loc="lower right", fontsize=8.5)
    title(fig, "Under DP, the simpler model wins: DP logistic regression keeps accuracy even at ε = 1",
          "Pima, mean ± 95% CI over 10 splits · dashed = non-private references")
    fig.savefig(f"{out}/fig4_central_dp_frontier.png")
    plt.close(fig)


def fig_lira(R, out):
    z = np.load(f"{R}/exp_lira/roc_curves.npz")
    s = pd.read_csv(f"{R}/exp_lira/summary.csv").set_index("model")
    grid = z["fpr_grid"]
    show = [("RandomForest", "Random Forest"), ("MLP", "MLP (non-private)"), ("MLP-matched", "MLP, 30-epoch recipe"),
            ("DP-MLP clip-only", "Clip only (ε = ∞)"), ("DP-MLP eps=4", "DP-SGD ε = 4"), ("LogReg", "Logistic Regression")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for i, (k, name) in enumerate(show):
        ax.plot(grid, z[k.replace(" ", "_")], color=CAT[i], lw=1.8, label=name)
    ax.plot(grid, grid, color=GREY, lw=1, ls=(0, (4, 3)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-4, 1); ax.set_ylim(1e-4, 1)
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.legend(fontsize=7.6, loc="lower right")
    ax2 = axes[1]
    order = ["RandomForest", "MLP", "MLP-L2", "MLP-matched", "DP-MLP clip-only", "DP-MLP eps=8", "DP-MLP eps=4",
             "DP-MLP eps=1", "DP-LR eps=4", "LogReg"]
    d = s.loc[order]
    yy = np.arange(len(order))[::-1]
    cols = [CAT[0] if ("eps=" in m) else GREY for m in order]
    ax2.barh(yy, d["lira_tpr_at_1pct"], color=cols, height=0.62)
    ax2.axvline(0.01, color=INK2, lw=1, ls=(0, (4, 3)))
    ax2.text(0.0105, yy[0] + 0.45, "chance (1%)", fontsize=8, color=INK2)
    for yi, v in zip(yy, d["lira_tpr_at_1pct"]):
        ax2.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax2.set_yticks(yy, order, fontsize=8.5)
    ax2.set_xlabel("TPR at 1% FPR (members exposed with high confidence)")
    ax2.grid(axis="y", visible=False)
    title(fig, "A stronger attack (LiRA) confirms the ranking and exposes members at low false-positive rates",
          "Pima, 64 models per configuration, each trained on a random half; 49,152 pooled decisions per row · dashed diagonal = random guess")
    fig.savefig(f"{out}/fig2_lira.png")
    plt.close(fig)


def fig_ldp(R, out):
    l = pd.read_csv(f"{R}/exp_ldp/summary.csv")
    m = pd.read_csv(f"{R}/exp_main/summary.csv").set_index("model")
    eps_c = [0.5, 1, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for ax, col, yl in [(axes[0], "roc_auc", "Test ROC-AUC (diabetes)"), (axes[1], "f1", "Test F1 (diabetes)")]:
        c = m.loc[[f"DP-LR eps={e:g}" for e in eps_c]]
        ax.errorbar(eps_c, c[f"{col}_mean"], yerr=c[f"{col}_ci95"], fmt="-o", color=CAT[0], ms=5, lw=2,
                    capsize=0, mec="white", label="Central DP (DP-SGD logistic regression)")
        for i, (mode, name) in enumerate([("label", "Local DP, label only (randomised response)"),
                                          ("full", "Local DP, all 9 attributes (RR + Laplace)")]):
            d = l[l["mode"] == mode].sort_values("epsilon")
            ax.errorbar(d["epsilon"], d[f"{col}_mean"], yerr=d[f"{col}_ci95"], fmt="-o", color=CAT[i + 1], ms=5,
                        lw=2, capsize=0, mec="white", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"])
        ax.set_xlabel("Privacy budget ε per patient")
        ax.set_ylabel(yl)
        ref = m.loc["LogReg", f"{col}_mean"]
        ax.axhline(ref, color=GREY, lw=1, ls=(0, (4, 3)))
    axes[0].axhline(0.5, color=GREY, lw=0.8)
    axes[0].text(0.55, 0.505, "no skill", fontsize=8, color=INK2, ha="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    title(fig, "Local DP costs far more utility than central DP for the same ε",
          "Pima, mean ± 95% CI over 10 splits · dashed = non-private logistic regression · local DP: each patient perturbs their own record")
    fig.savefig(f"{out}/fig7_local_vs_central.png")
    plt.close(fig)


def fig_kanon(R, out):
    s = pd.read_csv(f"{R}/exp_kanon/summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    for i, model in enumerate(["RandomForest", "LogReg"]):
        d = s[s["model"] == model].sort_values("k")
        for ax, col, yl in [(axes[0], "shadow_auc", "Shadow-model attack AUC"), (axes[1], "test_acc", "Test accuracy")]:
            ax.errorbar(d["k"], d[f"{col}_mean"], yerr=d[f"{col}_ci95"], fmt="-o", color=CAT[i], ms=5, lw=2,
                        capsize=0, mec="white", label="Random Forest" if model == "RandomForest" else "Logistic Regression")
            ax.set_ylabel(yl)
    d = s[s["model"] == "LogReg"].sort_values("k")
    axes[2].plot(d["k"], d["ncp_mean"], "-o", color=INK2, ms=5, lw=2)
    axes[2].set_ylabel("Information loss on QIs (NCP)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 25, 50], ["none", "2", "5", "10", "25", "50"])
        ax.set_xlabel("k (Mondrian k-anonymity)")
        ax.minorticks_off()
    axes[0].axhline(0.5, color=GREY, lw=1, ls=(0, (4, 3)))
    axes[0].legend(fontsize=8, loc="center right")
    title(fig, "k-anonymising the training data does not stop the model from leaking membership",
          "Pima, mean ± 95% CI over 10 splits · quasi-identifiers: Age, Pregnancies, BMI")
    fig.savefig(f"{out}/fig8_kanonymity.png")
    plt.close(fig)


def fig_compressive(R, out):
    s = pd.read_csv(f"{R}/exp_compressive/summary.csv").set_index("config")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    colors = {"none": GREY, "pca": CAT[1], "dca": CAT[0], "pdca": CAT[6]}
    offs = {"raw (8-D)": (6, -10), "PCA m=4": (6, 4), "PCA m=3": (6, -11), "PCA m=2": (6, 2), "PCA m=1": (6, 7),
            "DCA m=1": (6, 4), "PDCA m=1 b=1": (6, -10), "PDCA m=1 b=3": (7, 5), "PDCA m=1 b=10": (7, -9)}
    for cfg, r in s.iterrows():
        ax.errorbar(r["attr_auc_mean"], r["roc_auc_mean"], xerr=r["attr_auc_ci95"], yerr=r["roc_auc_ci95"],
                    fmt="o", color=colors[r["method"]], ms=7, mec="white", mew=1.2, elinewidth=1, capsize=0, zorder=3)
        dx, dy = offs.get(cfg, (6, 3))
        ax.annotate(cfg.replace(" b=", " β="), (r["attr_auc_mean"], r["roc_auc_mean"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.8, color=INK, ha="left" if dx > 0 else "right")
    ax.axvline(0.5, color=GREY, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("Attacker's AUC for inferring age ≥ 35 from Z  (← more private)")
    ax.set_ylabel("Diabetes ROC-AUC from Z  (↑ more useful)")
    ax2 = axes[1]
    order = list(s.index)
    yy = np.arange(len(order))[::-1]
    ax2.barh(yy, s.loc[order, "recon_r2_mean"], color=[colors[s.loc[o, "method"]] for o in order], height=0.6)
    for yi, v in zip(yy, s.loc[order, "recon_r2_mean"]):
        ax2.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
    ax2.set_yticks(yy, [o.replace(" b=", " β=") for o in order], fontsize=8.3)
    ax2.set_xlabel("R² of reconstructing all 8 features from Z")
    ax2.set_xlim(0, 1.1)
    ax2.grid(axis="y", visible=False)
    title(fig, "One discriminant component keeps the diabetes signal and hides most of the record",
          "Pima, mean over 10 splits (± 95% CI on the left) · grey raw, orange PCA, blue DCA, violet privacy-aware DCA")
    fig.savefig(f"{out}/fig9_compressive.png")
    plt.close(fig)


def fig_scaling(R, out):
    s = pd.read_csv(f"{R}/exp_scaling/summary.csv")
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    np_ = s[s["model"] == "LogReg (non-private)"].sort_values("n")
    ax.errorbar(np_["n"], np_["roc_auc_mean"], yerr=np_["roc_auc_ci95"], fmt="-o", color=GREY, lw=2, ms=5,
                capsize=0, mec="white", label="non-private logistic regression")
    co = s[s["model"] == "DP-LR clip-only (no noise)"].sort_values("n")
    ax.errorbar(co["n"], co["roc_auc_mean"], yerr=co["roc_auc_ci95"], fmt="--", color=INK2, lw=1.4,
                capsize=0, label="same DP optimiser, no noise (ε = ∞)")
    for i, e in enumerate([4.0, 1.0, 0.5]):
        d = s[(s["model"] == "DP-LR") & (s["epsilon"] == e)].sort_values("n")
        ax.errorbar(d["n"], d["roc_auc_mean"], yerr=d["roc_auc_ci95"], fmt="-o", color=CAT[i], lw=2, ms=5,
                    capsize=0, mec="white", label=f"DP-SGD logistic regression, ε = {e:g}")
    ax.set_xscale("log")
    ax.set_xlabel("Training records n (CDC BRFSS 2015)")
    ax.set_ylabel("Test ROC-AUC (20,000 held-out records)")
    ax.legend(fontsize=8.2, loc="lower right")
    title(fig, "DP gets cheaper with more data: from n = 25,000 the noise costs nothing",
          "CDC Diabetes Health Indicators, mean ± 95% CI over 3 seeds · δ = 1/(10 n)")
    fig.savefig(f"{out}/fig5_scaling.png")
    plt.close(fig)


def fig_sensitivity(R, out):
    s = pd.read_csv(f"{R}/exp_sensitivity/summary.csv")
    base = pd.read_csv(f"{R}/exp_main/summary.csv").set_index("model").loc["DP-MLP eps=4"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    c = s[s["vary"] == "clip"].sort_values("clip")
    c = pd.concat([c, pd.DataFrame([{"clip": 1.0, "test_acc_mean": base["test_acc_mean"], "test_acc_ci95": base["test_acc_ci95"]}])]).sort_values("clip")
    axes[0].errorbar(c["clip"], c["test_acc_mean"], yerr=c["test_acc_ci95"], fmt="-o", color=CAT[0], lw=2, ms=5, capsize=0, mec="white")
    axes[0].set_xscale("log"); axes[0].set_xticks([0.1, 0.5, 1, 2, 5], ["0.1", "0.5", "1", "2", "5"]); axes[0].minorticks_off()
    axes[0].set_xlabel("Clipping norm C (30 epochs)"); axes[0].set_ylabel("Test accuracy at ε = 4")
    e = s[s["vary"] == "epochs"].sort_values("epochs")
    e = pd.concat([e, pd.DataFrame([{"epochs": 30, "test_acc_mean": base["test_acc_mean"], "test_acc_ci95": base["test_acc_ci95"]}])]).sort_values("epochs")
    axes[1].errorbar(e["epochs"], e["test_acc_mean"], yerr=e["test_acc_ci95"], fmt="-o", color=CAT[0], lw=2, ms=5, capsize=0, mec="white")
    axes[1].set_xlabel("Training epochs (C = 1); σ re-calibrated each time")
    for ax in axes:
        ax.axvline(1.0 if ax is axes[0] else 30, color=GREY, lw=1, ls=(0, (4, 3)))
    axes[0].text(1.05, axes[0].get_ylim()[0] + 0.005, "setting used", fontsize=8, color=INK2)
    title(fig, "DP-SGD is sensitive to its settings: at the same ε, accuracy varies by 11 points",
          "Pima, DP-SGD MLP at ε = 4, δ = 10⁻⁵ · mean ± 95% CI over 10 splits · sensitivity analysis, not tuning")
    fig.savefig(f"{out}/fig6_sensitivity.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (fig_decomposition, fig_frontier, fig_lira, fig_ldp, fig_kanon, fig_compressive, fig_scaling, fig_sensitivity):
        try:
            f(a.results, a.out)
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, "(missing", e.filename, ")")


if __name__ == "__main__":
    main()
