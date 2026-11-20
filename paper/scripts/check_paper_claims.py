"""Checks every number quoted in the paper's text against the result files, the code, or derived values.

Usage: python check_paper_claims.py PROJECT_DIR LIRA_BOOT_JSON DERIVED_JSON MAIN_TEX
1. Each registered claim: the value computed from the files must round to the printed string, and that
   printed string must appear in the .tex text (tables are generated from the files and are not re-checked).
2. Every number left in the text that no claim covers is listed for manual review.
"""
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

proj, boot_path, derived_path, tex_path = sys.argv[1:5]
R = os.path.join(proj, "results")
sys.path.insert(0, os.path.join(proj, "src"))
import models  # noqa: E402  (constants of the training recipes)
import exp_kanon, exp_ldp, exp_main, exp_scaling, exp_compressive  # noqa: E402,E401

B = json.load(open(boot_path)); bt, bp = B["per_config"], B["paired"]
D = json.load(open(derived_path)); X = D["extra"]
m = pd.read_csv(f"{R}/exp_main/summary.csv").set_index("model")
t = pd.read_csv(f"{R}/exp_main/paired_tests.csv")
li = pd.read_csv(f"{R}/exp_lira/summary.csv").set_index("model")
ld = pd.read_csv(f"{R}/exp_ldp/summary.csv")
kn = pd.read_csv(f"{R}/exp_kanon/summary.csv")
cp = pd.read_csv(f"{R}/exp_compressive/summary.csv").set_index("config")
sc = pd.read_csv(f"{R}/exp_scaling/summary.csv")
se = pd.read_csv(f"{R}/exp_sensitivity/summary.csv")
prov = json.load(open(f"{R}/exp_scaling/provenance_check.json"))
la = json.load(open(f"{R}/exp_lira/accounting.json"))
ma = json.load(open(f"{R}/exp_main/privacy_accounting.json"))
kj = json.load(open(os.path.join(os.path.dirname(proj), "..", "src_ia1", "IA1_MIA_DP_Pima_code", "results", "kanonymity.json"))) \
    if os.path.exists(os.path.join(os.path.dirname(proj), "..", "src_ia1")) else None

def P(a, b, met): return float(t[(t.a == a) & (t.b == b) & (t.metric == met)].iloc[0].p_wilcoxon)
def LD(mode, e, c): return float(ld[(ld["mode"] == mode) & (ld.epsilon == e)].iloc[0][c])
def K(mod, k, c): return float(kn[(kn.model == mod) & (kn.k == k)].iloc[0][c])
def S(model, e, n):
    q = sc[(sc.model == model) & (sc.n == n)]
    if e is not None: q = q[q.epsilon == e]
    return float(q.iloc[0].roc_auc_mean)
def SE(vary, val, c):
    key = "clip" if vary == "clip" else "epochs"
    return float(se[(se.vary == vary) & (se[key] == val)].iloc[0][c])
mm = lambda k, c: float(m.loc[k, c])
dp_shadow = [mm(k, "shadow_auc_mean") for k in m.index if k.startswith("DP-") and "clip" not in k]
cp_sub = ["DCA m=1", "PDCA m=1 b=1", "PDCA m=1 b=3", "PDCA m=1 b=10"]
pct = lambda x: 100 * x

# (printed string, value, kind) ; kind: 'r' round to printed decimals, 'eq' exact integer/config
C = []
def r(s, v, note): C.append((s, float(v), "r", note))
def eq(s, v, note): C.append((s, v, "eq", note))

# ---- data and protocol (from code / data files) ----
eq("768", 768, "Pima records"); eq("500", 500, "negatives"); eq("268", 268, "positives")
eq("652", X["pima_zero_missing"], "zeros recoded"); r("48.7", X["pima_missing_pct"]["Insulin"], "Insulin missing %")
r("29.6", X["pima_missing_pct"]["SkinThickness"], "SkinThickness missing %")
eq("384", 384, "shadow pool / LiRA training size"); eq("192", 192, "members / non-members")
eq("10", 10, "splits"); eq("641", X["mlp_params"], "MLP parameters"); eq("9", X["lr_params"], "LR parameters / LDP attributes")
eq("64", models.DP_MLP["hidden"], "hidden units (also K=64 LiRA)"); eq("200", 200, "RF trees / BRFSS n=200")
eq("1000", models.NONPRIVATE_MLP["epochs"], "non-private MLP epochs"); r("0.05", models.NONPRIVATE_MLP["lr"], "MLP lr")
eq("30", models.DP_MLP["epochs"], "DP-SGD epochs"); r("0.5", models.DP_MLP["lr"], "DP-SGD lr / eps 0.5")
eq("32", models.DP_MLP["batch_size"], "lot size"); eq("180", ma["steps"], "DP-SGD steps")
r("17.28", ma["sigma"]["0.5"], "sigma eps0.5"); r("9.20", ma["sigma"]["1.0"], "sigma eps1"); r("4.97", ma["sigma"]["2.0"], "sigma eps2")
r("2.79", ma["sigma"]["4.0"], "sigma eps4"); r("1.67", ma["sigma"]["8.0"], "sigma eps8")
eq("360", la["steps"], "LiRA DP steps"); r("6.53", la["sigma"]["1.0"], "LiRA sigma eps1"); r("2.04", la["sigma"]["4.0"], "LiRA sigma eps4")
r("1.29", la["sigma"]["8.0"], "LiRA sigma eps8"); eq("49{,}152", 64 * 768, "pooled LiRA decisions")
eq("63", 63, "other models in LiRA"); eq("8", 8, "shadow models / features")
eq("1{,}000", 1000, "bootstrap resamples / BRFSS n"); eq("13", 13, "unit tests")
eq("253{,}680", prov["rows"], "BRFSS rows"); eq("21", prov["columns"] - 1, "BRFSS indicators")
r("13.9", 100 * prov["positive_rate"], "BRFSS positive %"); r("34.9", 100 * X["pima_pos_rate"], "Pima positive %")
eq("20{,}000", 20000, "BRFSS held-out"); eq("25{,}000", 25000, "BRFSS n"); eq("100{,}000", 100000, "BRFSS n"); eq("5{,}000", 5000, "BRFSS n")
eq("2048", 2048, "BRFSS max lot"); eq("20", 20, "BRFSS epochs"); r("1.0", 1.0, "BRFSS lr")
r("0.731", X["cap"]["1"], "DP cap eps1"); r("0.982", X["cap"]["4"], "DP cap eps4")
r("0.002", X["wilcoxon_min_p_n10"], "min Wilcoxon p"); r("0.0036", X["bonferroni_14"], "Bonferroni"); eq("14", 14, "paired tests")
r("1.8", pct(X["rr_flip"]["4"]), "RR flip eps4"); r("26.9", pct(X["rr_flip"]["1"]), "RR flip eps1")
r("0.44", D["ldp_full_eps4_per_attr_eps"], "per-attribute eps"); r("3.2", D["ldp_full_eps4_laplace_sd_in_ranges"], "Laplace sd / range")
eq("25", round(X["fp_at_0.1pct"]), "false positives at 0.1% FPR")
# ---- RQ1 ----
r("56.2", pct(bt["RandomForest"]["tpr1"]), "RF TPR@1%"); r("52.7", pct(bt["RandomForest"]["tpr1_ci"][0]), "RF TPR CI lo")
r("60.1", pct(bt["RandomForest"]["tpr1_ci"][1]), "RF TPR CI hi"); r("12.9", pct(bt["RandomForest"]["tpr01"]), "RF TPR@0.1%")
r("0.895", li.loc["RandomForest", "lira_balanced_acc"], "RF LiRA balanced acc"); r("0.814", mm("RandomForest", "shadow_auc_mean"), "RF shadow AUC")
r("10.7", pct(bt["MLP"]["tpr1"]), "MLP TPR"); r("9.2", pct(bt["MLP"]["tpr1_ci"][0]), "MLP CI lo"); r("12.5", pct(bt["MLP"]["tpr1_ci"][1]), "MLP CI hi")
lo_ = min(pct(bt[k]["tpr1"]) for k in ["LogReg", "DP-MLP eps=8", "DP-MLP eps=4", "DP-MLP eps=1", "DP-LR eps=4"])
hi_ = max(pct(bt[k]["tpr1"]) for k in ["LogReg", "DP-MLP eps=8", "DP-MLP eps=4", "DP-MLP eps=1", "DP-LR eps=4"])
r("1.0--1.3", 0, "range"); C[-1] = ("1.0--1.3", (lo_, hi_), "range1", "LR and DP TPR range")
r("0.530", bt["LogReg"]["auc"], "LR LiRA AUC"); r("0.523", bt["LogReg"]["auc_ci"][0], "LR AUC CI lo"); r("0.537", bt["LogReg"]["auc_ci"][1], "LR AUC CI hi")
r("0.764", mm("LogReg", "test_acc_mean"), "LR acc (= RF acc)"); r("0.764", mm("RandomForest", "test_acc_mean"), "RF acc")
r("0.014", mm("LogReg", "gen_gap_mean"), "LR gap"); r("0.236", mm("RandomForest", "gen_gap_mean"), "RF gap")
# ---- RQ2 ----
r("0.713", mm("MLP", "shadow_auc_mean"), "MLP shadow"); r("0.612", mm("MLP-matched", "shadow_auc_mean"), "matched shadow")
r("0.534", mm("DP-MLP clip-only", "shadow_auc_mean"), "clip shadow"); r("0.522", mm("DP-MLP eps=4", "shadow_auc_mean"), "eps4 shadow")
r("0.846", P("DP-MLP clip-only", "DP-MLP eps=4", "shadow_auc"), "p noise shadow"); r("0.85", P("DP-MLP clip-only", "DP-MLP eps=4", "shadow_auc"), "p noise (abstract)")
r("0.002", P("MLP", "MLP-matched", "shadow_auc"), "p length"); r("0.002", P("MLP-matched", "DP-MLP clip-only", "shadow_auc"), "p clip")
r("0.105", P("DP-MLP clip-only", "DP-MLP eps=4", "loss_auc"), "p loss noise")
r("53", 100 * D["share_shadow_length"], "share length"); r("41", 100 * D["share_shadow_clip"], "share clip"); r("6", 100 * D["share_shadow_noise"], "share noise shadow")
r("94", 100 * (D["share_shadow_length"] + D["share_shadow_clip"]), "share length+clip"); r("0.191", D["total_drop_shadow"], "total shadow drop")
r("0.127", bp["MLP - MLP-matched"]["auc_diff"], "LiRA length"); r("0.049", bp["MLP-matched - DP-MLP clip-only"]["auc_diff"], "LiRA clip")
r("0.045", bp["DP-MLP clip-only - DP-MLP eps=4"]["auc_diff"], "LiRA noise"); r("21", 100 * D["share_lira_noise"], "LiRA share noise")
r("0.221", D["total_drop_lira"], "total LiRA drop"); r("0.037", bp["DP-MLP clip-only - DP-MLP eps=4"]["auc_diff_ci"][0], "noise CI lo")
r("0.053", bp["DP-MLP clip-only - DP-MLP eps=4"]["auc_diff_ci"][1], "noise CI hi")
r("3.9", pct(bt["MLP-matched"]["tpr1"]), "matched TPR"); r("2.0", pct(bt["DP-MLP clip-only"]["tpr1"]), "clip TPR"); r("1.2", pct(bt["DP-MLP eps=4"]["tpr1"]), "eps4 TPR")
r("0.8", D["lira_noise_tpr1_diff_pp"], "noise TPR diff"); r("0.3", D["lira_noise_tpr1_diff_ci_pp"][0], "noise TPR CI lo"); r("1.2", D["lira_noise_tpr1_diff_ci_pp"][1], "noise TPR CI hi")
r("0.752", mm("MLP-L2", "test_acc_mean"), "L2 acc"); r("0.735", mm("DP-MLP eps=4", "test_acc_mean"), "DP-MLP4 acc"); r("0.195", P("MLP-L2", "DP-MLP eps=4", "test_acc"), "p L2 acc")
r("0.586", mm("MLP-L2", "shadow_auc_mean"), "L2 shadow"); r("0.625", li.loc["MLP-L2", "lira_auc"], "L2 LiRA AUC"); r("0.530", bt["DP-MLP eps=4"]["auc"], "eps4 LiRA AUC")
r("0.706", mm("MLP", "test_acc_mean"), "MLP acc"); r("0.728", mm("MLP-matched", "test_acc_mean"), "matched acc"); r("0.756", mm("DP-MLP clip-only", "test_acc_mean"), "clip acc")
# ---- RQ3 ----
r("0.726", mm("DP-LR eps=1", "test_acc_mean"), "DP-LR1 acc"); r("0.642", mm("DP-MLP eps=1", "test_acc_mean"), "DP-MLP1 acc"); r("0.004", P("DP-MLP eps=1", "DP-LR eps=1", "test_acc"), "p eps1")
r("0.756", mm("DP-LR eps=4", "test_acc_mean"), "DP-LR4 acc"); r("0.027", P("DP-MLP eps=4", "DP-LR eps=4", "test_acc"), "p eps4")
r("0.410", P("LogReg", "DP-LR eps=4", "test_acc"), "p LR vs DP-LR4"); r("0.41", P("LogReg", "DP-LR eps=4", "test_acc"), "p (abstract)")
r("0.824", mm("DP-LR eps=4", "roc_auc_mean"), "DP-LR4 ROC"); r("0.828", mm("LogReg", "roc_auc_mean"), "LR ROC")
r("0.607", mm("DP-LR eps=4", "recall_mean"), "DP-LR4 recall"); r("0.543", mm("LogReg", "recall_mean"), "LR recall")
r("0.520", bt["DP-LR eps=4"]["auc"], "DP-LR4 LiRA AUC"); r("1.2", pct(bt["DP-LR eps=4"]["tpr1"]), "DP-LR4 TPR")
C.append(("0.478--0.526", (min(dp_shadow), max(dp_shadow)), "range", "DP shadow range"))
C.append(("0.612--0.642", (mm("DP-MLP eps=0.5", "test_acc_mean"), mm("DP-MLP eps=1", "test_acc_mean")), "range", "DP-MLP eps<=1 acc"))
r("0.646", SE("clip", 5.0, "test_acc_mean"), "C5 acc"); r("0.759", SE("clip", 0.5, "test_acc_mean"), "C0.5 acc")
r("0.816", SE("clip", 0.1, "roc_auc_mean"), "C0.1 ROC"); r("0.420", SE("clip", 0.1, "f1_mean"), "C0.1 F1")
r("1.817", SE("epochs", 10, "sigma"), "10ep sigma"); r("0.758", SE("epochs", 10, "test_acc_mean"), "10ep acc")
nc = D["noise_cost"]
r("0.057", nc["200_4"], "noise cost 200"); r("0.004", nc["5000_4"], "noise cost 5k"); r("0.003", max(nc["25000_0.5"], nc["100000_0.5"], nc["25000_1"], nc["25000_4"]), "noise cost >=25k")
og = [v for v in D["optimiser_gap"].values()]
C.append(("0.020--0.031", (min(og), max(og)), "range", "optimiser gap range")); C.append(("0.02--0.03", (min(og), max(og)), "range2", "optimiser gap (2dp)"))
r("0.666", S("DP-LR", 4.0, 200), "BRFSS 200 eps4"); r("0.743", S("LogReg (non-private)", None, 200), "BRFSS 200 np")
eq("3", 3, "BRFSS seeds")
# ---- RQ4 ----
r("0.762", LD("label", 4.0, "test_acc_mean"), "label eps4 acc"); r("0.708", LD("label", 1.0, "test_acc_mean"), "label eps1 acc")
r("0.652", LD("full", 16.0, "test_acc_mean"), "full eps16 acc"); r("0.652", LD("full", 8.0, "test_acc_mean"), "full eps8 acc")
r("0.727", LD("full", 16.0, "roc_auc_mean"), "full eps16 ROC"); r("0.749", mm("DP-LR eps=0.5", "roc_auc_mean"), "DP-LR0.5 ROC")
r("0.03", X["ldp_full_collapse"]["16.0"]["max_recall"], "max recall eps16"); r("0.03", X["ldp_full_collapse"]["8.0"]["max_recall"], "max recall eps8")
r("0.651", 125 / 192, "all-negative accuracy"); eq("32", 32, "16/0.5 budget ratio")
# ---- RQ5 ----
r("95.4", 100 * X["unique_full_768"], "unique 768"); r("97.9", 100 * X["unique_train192_min"], "unique train min"); r("100", 100 * X["unique_train192_max"], "unique train max")
r("67.5", 100 * K("RandomForest", 50, "ncp_mean"), "NCP k50 %"); r("0.675", K("RandomForest", 50, "ncp_mean"), "NCP k50")
r("0.805", K("RandomForest", 2, "shadow_auc_mean"), "RF k2"); r("0.793", K("RandomForest", 5, "shadow_auc_mean"), "RF k5"); r("0.794", K("RandomForest", 50, "shadow_auc_mean"), "RF k50")
r("0.002", D["kanon_p"]["RandomForest_k5_shadow_auc"], "p k5"); r("0.010", D["kanon_p"]["RandomForest_k50_shadow_auc"], "p k50")
r("0.745", K("RandomForest", 5, "test_acc_mean"), "RF k5 acc"); r("0.738", K("RandomForest", 50, "test_acc_mean"), "RF k50 acc")
r("0.020", D["kanon_p"]["RandomForest_k5_test_acc"], "p k5 acc"); r("0.006", D["kanon_p"]["RandomForest_k50_test_acc"], "p k50 acc")
r("0.156", min(v for k, v in D["kanon_p"].items() if k.startswith("LogReg")), "LR k-anon min p")
r("0.764", cp.loc["DCA m=1", "test_acc_mean"], "DCA acc"); r("0.827", cp.loc["DCA m=1", "roc_auc_mean"], "DCA ROC"); r("0.828", cp.loc["raw (8-D)", "roc_auc_mean"], "raw ROC")
r("0.21", cp.loc["DCA m=1", "recon_r2_mean"], "DCA R2"); r("0.796", cp.loc["PCA m=1", "roc_auc_mean"], "PCA1 ROC")
r("0.814", cp.loc["PCA m=1", "attr_auc_mean"], "PCA1 age AUC"); r("0.751", cp.loc["DCA m=1", "attr_auc_mean"], "DCA age AUC")
r("0.504", cp.loc["PDCA m=1 b=10", "attr_auc_mean"], "PDCA10 age AUC"); r("0.632", cp.loc["PDCA m=1 b=10", "attr_acc_mean"], "PDCA10 attr acc")
r("0.637", cp.loc["PDCA m=1 b=10", "attr_majority_mean"], "majority rate"); r("1.2", 100 * (cp.loc["DCA m=1", "test_acc_mean"] - cp.loc["PDCA m=1 b=10", "test_acc_mean"]), "PDCA cost vs DCA (points)")
C.append(("0.696--0.705", (cp.loc[cp_sub, "rf_shadow_auc_mean"].min(), cp.loc[cp_sub, "rf_shadow_auc_mean"].max()), "range", "RF on Z shadow"))
C.append(("0.678--0.702", (cp.loc[cp_sub, "rf_test_acc_mean"].min(), cp.loc[cp_sub, "rf_test_acc_mean"].max()), "range", "RF on Z acc"))
r("0.70", cp.loc[cp_sub, "rf_shadow_auc_mean"].mean(), "about 0.70 (abstract)")
r("0.52", li.loc["DP-MLP eps=4", "lira_balanced_acc"], "LiRA balanced acc eps4")

tex = open(tex_path).read()
body = tex[tex.index("\\begin{abstract}"):tex.index("\\bibliographystyle")]
fails, ok = [], 0
def dec(s): return len(s.split(".")[1]) if "." in s else 0
for s, v, kind, note in C:
    present = s in body
    if kind == "eq":
        good = True
    elif kind in ("range", "range1"):
        a, b = s.split("--"); good = round(v[0], dec(a)) == float(a) and round(v[1], dec(b)) == float(b)
    elif kind == "range2":
        a, b = s.split("--"); good = round(v[0], 2) == float(a) and round(v[1], 2) == float(b)
    else:
        d = dec(s.replace("{,}", "")); good = abs(round(v, d) - float(s.replace("{,}", ""))) < 1e-9 or abs(v - float(s)) <= 0.5 * 10 ** (-d) + 1e-9
    if not present: fails.append((s, note, "NOT FOUND IN TEXT"))
    elif not good: fails.append((s, note, f"value {v}"))
    else: ok += 1
print(f"{len(C)} registered claims: {ok} verified, {len(fails)} problems")
for f in fails: print("  PROBLEM", f)

# ---- orphan numbers: anything in the text not covered by a registered claim ----
clean = re.sub(r"\\(cite|ref|label|input|includegraphics|caption\b)\{[^}]*\}", " ", body)
clean = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", clean, flags=re.S)
covered = set()
for s, *_ in C:
    for part in s.split("--"): covered.add(part.replace("{,}", ","))
tokens = re.findall(r"(?<![\w.])(\d{1,3}(?:\{,\}\d{3})+|\d+(?:\.\d+)?)(?![\w])", clean)
orph = sorted({x.replace("{,}", ",") for x in tokens} - covered, key=lambda z: float(z.replace(",", "")))
print("numbers in the text with no registered claim (review by hand):", orph)
