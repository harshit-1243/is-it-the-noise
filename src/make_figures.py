"""Result figures (Figures 3-5) for the IA-1 report, built from results/ (run run_experiments.py first).
Figure 1 (workflow) is rendered from figures/workflow_diagram.html; Figure 2 is the console log.

Usage: python src/make_figures.py --results results --out figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def title(fig, main, sub, top=0.84):
    """Finding as the title, provenance as the subtitle; plots sit below `top`."""
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.text(0.01, 0.985, main, fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 0.985 - 0.075 * 4.2 / fig.get_figheight(), sub, fontsize=9.5, color=INK2, ha="left", va="top")


def fig_gap_vs_leakage(s: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    short = {"LogReg": "Logistic Regression", "RandomForest": "Random Forest", "MLP": "MLP (non-private)",
             "DP-MLP (clip only)": "DP-SGD, clip only (ε=∞)"}
    offsets = {"LogReg": (8, -14), "RandomForest": (-10, 8), "MLP": (-10, 8), "DP-MLP (clip only)": (8, 4)}
    for m, r in s.iterrows():
        dp = m.startswith("DP-MLP")
        c = BLUE if dp else INK2
        ax.errorbar(r["gen_gap_mean"], r["shadow_attack_auc_mean"], xerr=r["gen_gap_std"],
                    yerr=r["shadow_attack_auc_std"], fmt="o", ms=7, color=c, ecolor=c, elinewidth=1,
                    alpha=0.9, capsize=0, mec="white", mew=1.2, zorder=3)
        if m in short:
            dx, dy = offsets[m]
            ax.annotate(short[m], (r["gen_gap_mean"], r["shadow_attack_auc_mean"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=9, color=INK,
                        ha="right" if dx < 0 else "left")
    dp = s[s.index.str.contains("eps=")]
    ax.annotate("DP-SGD, ε = 0.5 to 8", (dp["gen_gap_mean"].max(), dp["shadow_attack_auc_mean"].mean()),
                xytext=(70, -4), textcoords="offset points", fontsize=9, color=BLUE, va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.axhline(0.5, color=GREY, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.33, 0.503, "random guess (AUC = 0.5)", color=INK2, fontsize=8.5, ha="right", va="bottom")
    ax.set_xlim(-0.02, 0.36)
    ax.set_ylim(0.42, 0.86)
    ax.set_xlabel("Generalisation gap (train accuracy − test accuracy)")
    ax.set_ylabel("Shadow-model attack AUC")
    title(fig, "The more a model overfits, the more membership it leaks",
          "Mean ± 1 s.d. over 5 random splits · Pima diabetes · 192 members vs 192 non-members per split")
    fig.savefig(os.path.join(out, "fig3_gap_vs_leakage.png"))
    plt.close(fig)


def fig_privacy_utility(s: pd.DataFrame, out: str):
    order = ["DP-MLP eps=0.5", "DP-MLP eps=1", "DP-MLP eps=2", "DP-MLP eps=4", "DP-MLP eps=8", "DP-MLP (clip only)"]
    labels = ["0.5", "1", "2", "4", "8", "∞\n(clip only)"]
    d = s.loc[order]
    x = np.arange(len(order))
    np_mlp, lr = s.loc["MLP"], s.loc["LogReg"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharex=True)
    panels = [
        (axes[0], "test_acc", "Test accuracy", (0.45, 0.82),
         [(np_mlp["test_acc_mean"], "non-private MLP"), (lr["test_acc_mean"], "non-private LogReg")]),
        (axes[1], "shadow_attack_auc", "Shadow-model attack AUC", (0.45, 0.82),
         [(np_mlp["shadow_attack_auc_mean"], "non-private MLP"), (0.5, "random guess")]),
    ]
    for ax, col, ylab, ylim, refs in panels:
        ax.errorbar(x, d[f"{col}_mean"], yerr=d[f"{col}_std"], fmt="-o", color=BLUE, ms=6, lw=2,
                    elinewidth=1, capsize=0, mec="white", mew=1.2, zorder=3)
        for v, name in refs:
            ax.axhline(v, color=GREY, lw=1, ls=(0, (4, 3)), zorder=1)
            if name == "random guess":   # below the line, right side (clear of the error bars)
                ax.text(x[-1] + 0.35, v - 0.004, f"{name} {v:.2f}", color=INK2, fontsize=8.5, ha="right", va="top")
            else:                        # above the line, left side (data is low there)
                ax.text(-0.22, v + 0.004, f"{name} {v:.2f}", color=INK2, fontsize=8.5, ha="left", va="bottom")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Privacy budget ε  (δ = 10⁻⁵)")
        ax.set_xlim(-0.3, len(order) - 0.6)
    title(fig, "DP-SGD at ε = 4–8 keeps accuracy and removes most of the leakage",
          "DP-MLP, mean ± 1 s.d. over 5 splits · dashed lines = references")
    fig.savefig(os.path.join(out, "fig5_privacy_utility.png"))
    plt.close(fig)


def fig_loss_distributions(scores: dict, out: str):
    panels = [("RandomForest", "Random Forest"), ("MLP", "MLP (non-private)"), ("DP-MLP eps=4", "DP-MLP, ε = 4")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)
    bins = np.linspace(-7, 1.5, 35)
    for ax, (key, name) in zip(axes, panels):
        li = np.log10(np.maximum(-scores[f"{key}|loss_in"], 1e-7))
        lo = np.log10(np.maximum(-scores[f"{key}|loss_out"], 1e-7))
        ax.hist(li, bins=bins, color=BLUE, alpha=0.55, label="members (training set)")
        ax.hist(lo, bins=bins, color=ORANGE, alpha=0.55, label="non-members (held out)")
        ax.axvline(np.median(li), color=BLUE, lw=1.5)
        ax.axvline(np.median(lo), color=ORANGE, lw=1.5)
        ax.set_title(name, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("log₁₀ per-record loss")
    axes[0].set_ylabel("Records")
    h, l = axes[0].get_legend_handles_labels()
    title(fig, "Members get much lower loss than non-members; DP-SGD closes most of that gap",
          "Per-record cross-entropy loss, split seed 0 · vertical lines = medians · this gap is what the attacker exploits", top=0.78)
    fig.legend(h, l, frameon=False, fontsize=8.5, ncol=2, loc="upper right", bbox_to_anchor=(0.995, 0.87))
    fig.savefig(os.path.join(out, "fig4_loss_distributions.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = pd.read_csv(os.path.join(a.results, "summary.csv"), index_col=0)
    scores = dict(np.load(os.path.join(a.results, "scores_seed0.npz")))
    fig_gap_vs_leakage(s, a.out)
    fig_privacy_utility(s, a.out)
    fig_loss_distributions(scores, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
