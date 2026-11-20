"""Re-identification risk check (k-anonymity) on quasi-identifiers of the Pima dataset.

Shows that the raw table is not anonymous even without names, and how much
generalisation is needed before every record hides in a group of size >= k.
This motivates protecting the *model* (DP) and not only the *data*.

Usage: python src/kanon_check.py --data data/pima_diabetes.csv --out results
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

QUASI_IDENTIFIERS = ["Age", "Pregnancies", "BMI"]


def class_sizes(df: pd.DataFrame, cols) -> pd.Series:
    return df.groupby(cols, dropna=False)[cols[0]].transform("size")


def report(df: pd.DataFrame, cols, label: str) -> dict:
    k = class_sizes(df, cols)
    return {
        "scheme": label,
        "k (smallest group)": int(k.min()),
        "unique records": int((k == 1).sum()),
        "unique %": round(100 * float((k == 1).mean()), 1),
        "records in groups < 5": int((k < 5).sum()),
        "groups": int(df.groupby(cols, dropna=False).ngroups),
    }


def generalise(df: pd.DataFrame, age_w: int, bmi_w: float, preg_cap: int) -> pd.DataFrame:
    g = df.copy()
    g["Age"] = (g["Age"] // age_w) * age_w
    g["BMI"] = np.floor(g["BMI"] / bmi_w) * bmi_w
    g["Pregnancies"] = g["Pregnancies"].clip(upper=preg_cap)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pima_diabetes.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    rows = [report(df, QUASI_IDENTIFIERS, "Raw (exact Age, Pregnancies, BMI)")]
    for age_w, bmi_w, cap in [(5, 2.5, 8), (10, 5, 6), (20, 10, 4)]:
        rows.append(report(generalise(df, age_w, bmi_w, cap), QUASI_IDENTIFIERS,
                           f"Age/{age_w}y, BMI/{bmi_w:g}, Pregnancies capped at {cap}+"))
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, "kanonymity.json"), "w"), indent=2)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
