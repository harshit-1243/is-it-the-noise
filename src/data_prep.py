"""Data loading, auditing and splitting for the Pima Indians Diabetes dataset.

Dataset: Pima Indians Diabetes Database (NIDDK, via UCI ML Repository / Kaggle
"uciml/pima-indians-diabetes-database", CC0). 768 rows, 8 numeric features,
binary target `Outcome` (1 = diabetes within 5 years).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
# A value of 0 is physiologically impossible for these columns, so the
# dataset uses 0 as a hidden "missing" code.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert list(df.columns) == FEATURES + [TARGET], "unexpected schema"
    return df


def audit(df: pd.DataFrame) -> dict:
    """Preliminary data-quality audit (recorded in the preparation log)."""
    zeros = {c: int((df[c] == 0).sum()) for c in ZERO_AS_MISSING}
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_values": int(df.isna().sum().sum()),
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "positive_rate": round(float(df[TARGET].mean()), 4),
        "zero_coded_missing": zeros,
        "zero_coded_missing_pct": {c: round(100 * v / len(df), 1) for c, v in zeros.items()},
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Recode impossible zeros as NaN. Imputation happens later, inside each
    model pipeline, so statistics are learnt only from that model's training data."""
    out = df.copy()
    out[ZERO_AS_MISSING] = out[ZERO_AS_MISSING].replace(0, np.nan)
    return out


def make_splits(df: pd.DataFrame, seed: int) -> dict:
    """Split for a membership-inference experiment.

    50% shadow pool  -> attacker's own data from the same distribution
    25% target_in    -> target model's training set (members)
    25% target_out   -> held-out set (non-members, also used as test set)
    All splits are stratified on the label.
    """
    X, y = df[FEATURES], df[TARGET].to_numpy()
    X_tgt, X_sh, y_tgt, y_sh = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    X_in, X_out, y_in, y_out = train_test_split(X_tgt, y_tgt, test_size=0.5, stratify=y_tgt, random_state=seed)
    return {
        "shadow": (X_sh.reset_index(drop=True), y_sh),
        "target_in": (X_in.reset_index(drop=True), y_in),
        "target_out": (X_out.reset_index(drop=True), y_out),
    }
