"""Compare a fresh rerun of all experiments with the shipped result files (numeric columns, max abs diff)."""
import sys, os
import numpy as np, pandas as pd
a, b = sys.argv[1], sys.argv[2]
worst = 0.0
for exp in sorted(os.listdir(b)):
    for f in ("summary.csv", "paired_tests.csv", "runs.csv"):
        pa, pb = os.path.join(a, exp, f), os.path.join(b, exp, f)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        A, B = pd.read_csv(pa), pd.read_csv(pb)
        assert A.shape == B.shape, (exp, f, A.shape, B.shape)
        num = A.select_dtypes("number").columns
        d = np.nanmax(np.abs(A[num].to_numpy(float) - B[num].to_numpy(float))) if len(num) else 0.0
        nonnum = [c for c in A.columns if c not in num]
        same_labels = all((A[c].astype(str) == B[c].astype(str)).all() for c in nonnum)
        worst = max(worst, d)
        print(f"{exp:16s} {f:16s} rows={len(A):4d} max|diff|={d:.2e} labels_match={same_labels}")
print("WORST", worst)
