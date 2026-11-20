"""Privacy-preserving data mining (Module 4): Mondrian multidimensional k-anonymity
(LeFevre, DeWitt and Ramakrishnan, 2006) on the quasi-identifiers Age, Pregnancies and BMI.

Training data: recursively split the quasi-identifier space at the median of the widest
(normalised) dimension while both halves keep >= k records. Each leaf is an equivalence class;
its records' quasi-identifiers are replaced by the leaf's range midpoints (generalisation).
New records (queries at prediction time) are routed down the same tree and generalised to the
leaf they fall in, so the deployed model always sees generalised quasi-identifiers.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# column positions after SimpleImputer (order of data_prep.FEATURES)
QI_COLS = (7, 0, 5)   # Age, Pregnancies, BMI


class Mondrian(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 5, qi_cols=QI_COLS):
        self.k = k
        self.qi_cols = qi_cols

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Q = X[:, list(self.qi_cols)]
        self.span_ = np.maximum(Q.max(0) - Q.min(0), 1e-9)
        self.nodes_ = []          # (dim, threshold, left, right) or leaf (-1, midpoints, sizes)
        self.leaf_sizes_ = []
        self.root_ = self._build(Q, np.arange(len(Q)))
        return self

    def _build(self, Q, idx):
        sub = Q[idx]
        if len(idx) >= 2 * self.k:
            widths = (sub.max(0) - sub.min(0)) / self.span_
            for dim in np.argsort(-widths):
                if widths[dim] <= 0:
                    break
                vals = sub[:, dim]
                thr = np.median(vals)
                left, right = idx[vals <= thr], idx[vals > thr]
                if len(left) >= self.k and len(right) >= self.k:
                    node = len(self.nodes_)
                    self.nodes_.append(None)
                    l_id, r_id = self._build(Q, left), self._build(Q, right)
                    self.nodes_[node] = ("split", int(dim), float(thr), l_id, r_id)
                    return node
        node = len(self.nodes_)
        mids = (sub.min(0) + sub.max(0)) / 2.0
        self.nodes_.append(("leaf", mids, len(idx)))
        self.leaf_sizes_.append(len(idx))
        return node

    def _route(self, q):
        n = self.root_
        while self.nodes_[n][0] == "split":
            _, dim, thr, l_id, r_id = self.nodes_[n]
            n = l_id if q[dim] <= thr else r_id
        return n

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        cols = list(self.qi_cols)
        for i in range(len(X)):
            X[i, cols] = self.nodes_[self._route(X[i, cols])][1]
        return X

    # ---- reporting ---------------------------------------------------------- #
    @property
    def min_class_size_(self):
        return int(min(self.leaf_sizes_))

    @property
    def n_classes_(self):
        return int(len(self.leaf_sizes_))

    def ncp(self, X):
        """Normalised certainty penalty: mean generalised range / full range (0 = no loss, 1 = full suppression)."""
        X = np.asarray(X, dtype=float)
        Q = X[:, list(self.qi_cols)]
        lo = {}
        for i, q in enumerate(Q):
            lo.setdefault(self._route(q), []).append(i)
        pen = []
        for leaf, rows in lo.items():
            sub = Q[rows]
            pen += [((sub.max(0) - sub.min(0)) / self.span_).mean()] * len(rows)
        return float(np.mean(pen))
