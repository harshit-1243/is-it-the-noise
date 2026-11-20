"""Extra numbers computed for the paper from the user's own code and data (not in the shipped results files)."""
import sys, json
import numpy as np, pandas as pd
proj = sys.argv[1]
sys.path.insert(0, proj + '/src')
from data_prep import clean, load_dataset, make_splits
from ppdm import Mondrian
from sklearn.impute import SimpleImputer
out = {}
raw = load_dataset(proj + '/data/pima_diabetes.csv'); df = clean(raw)
# 1) QI uniqueness of the full table (re-derives IA-1's 95.4%) and of each 192-record training set
qi = ['Age', 'Pregnancies', 'BMI']
def uniq(d):
    c = d.groupby(qi, dropna=False)[qi[0]].transform('size')
    return float((c == 1).mean())
out['unique_full_768'] = uniq(raw)
u_tr, u_tr_k5 = [], []
for s in range(10):
    S = make_splits(df, s); X_in = S['target_in'][0]
    u_tr.append(uniq(X_in))
    imp = SimpleImputer(strategy='median').fit(X_in); Xi = imp.transform(X_in)
    m = Mondrian(k=5).fit(Xi); Xa = pd.DataFrame(m.transform(Xi), columns=X_in.columns)
    u_tr_k5.append(uniq(Xa))
out['unique_train192_mean'] = float(np.mean(u_tr)); out['unique_train192_min'] = float(np.min(u_tr)); out['unique_train192_max'] = float(np.max(u_tr))
out['unique_train192_after_k5_max'] = float(np.max(u_tr_k5))
# 2) full-record LDP collapse
r = pd.read_csv(proj + '/results/exp_ldp/runs.csv'); f = r[r['mode'] == 'full']
coll = {}
for e, g in f.groupby('epsilon'):
    single = int(((g.recall == 0) | (g.recall >= 0.98)).sum())
    coll[str(e)] = {'splits_recall_0_or_ge_0.98': single, 'max_recall': float(g.recall.max())}
out['ldp_full_collapse'] = coll
# 3) parameter counts and DP-SGD steps
out['mlp_params'] = 8 * 64 + 64 + 64 + 1; out['lr_params'] = 8 + 1
acc = json.load(open(proj + '/results/exp_main/privacy_accounting.json')); out['main_q'] = acc['q']; out['main_steps'] = acc['steps']
out['main_sigma'] = acc['sigma']
la = json.load(open(proj + '/results/exp_lira/accounting.json')); out['lira_q'] = la['q']; out['lira_steps'] = la['steps']; out['lira_sigma'] = la['sigma']
# 4) theoretical cap e^eps/(1+e^eps)
out['cap'] = {str(e): float(np.exp(e) / (1 + np.exp(e))) for e in (0.5, 1, 2, 4, 8)}
# 5) RR flip rates
out['rr_flip'] = {str(e): float(1 / (1 + np.exp(e))) for e in (1, 4)}
# 6) Pima class balance and missingness
out['pima_pos_rate'] = float(raw.Outcome.mean()); out['pima_zero_missing'] = int(df.isna().sum().sum())
out['pima_missing_pct'] = {c: round(100 * float(df[c].isna().mean()), 1) for c in ['Insulin', 'SkinThickness']}
# 7) LiRA pooled decisions
lr = pd.read_csv(proj + '/results/exp_lira/summary.csv'); out['lira_n_scores'] = sorted(set(lr.n_scores.astype(int)))
# 8) false positives behind TPR at 0.1% FPR: non-member decisions per config ~ 64*384
out['lira_nonmember_decisions'] = 64 * 384; out['fp_at_0.1pct'] = 64 * 384 * 0.001
# 9) Wilcoxon minimum attainable two-sided p with n=10
from scipy import stats
out['wilcoxon_min_p_n10'] = float(stats.wilcoxon(np.arange(1, 11), np.zeros(10)).pvalue)
out['bonferroni_14'] = 0.05 / 14
# 10) RF / DP shadow CI from summary
m = pd.read_csv(proj + '/results/exp_main/summary.csv').set_index('model')
out['n_seeds_main'] = int(m.n_seeds.iloc[0])
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else 'extra_checks.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
