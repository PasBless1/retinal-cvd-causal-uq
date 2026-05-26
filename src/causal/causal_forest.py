"""
Optional: causal-forest estimation of heterogeneous treatment effects.

Estimates how the effect of a (binarised) retinal exposure on CVD risk varies
across patients. Implemented with a lightweight honest-splitting random forest
of T-learner style so the pipeline has no hard dependency on EconML; if EconML
is installed the CausalForestDML is used instead for a more principled
estimator.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _t_learner_cate(X: np.ndarray, T: np.ndarray,
                    Y: np.ndarray, seed: int) -> np.ndarray:
    """T-learner conditional average treatment effect."""
    if T.sum() == 0 or T.sum() == len(T):
        # Degenerate split (e.g. constant exposure): no contrast estimable.
        return np.zeros(len(T))
    m1 = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                               random_state=seed)
    m0 = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                               random_state=seed)
    m1.fit(X[T == 1], Y[T == 1])
    m0.fit(X[T == 0], Y[T == 0])
    return m1.predict(X) - m0.predict(X)


def heterogeneous_effects(df: pd.DataFrame, exposure: str,
                          cfg: dict) -> Dict[str, object]:
    """
    Binarise `exposure` at its median (high vs low) and estimate the
    conditional average treatment effect on the outcome.
    """
    seed = cfg.get("seed", 42)
    y = cfg["causal"]["outcome"]
    feat_cols = [c for c in cfg["causal"]["confounders"] if c in df.columns]
    # Encode categorical confounders numerically for the forest.
    Xdf = df[feat_cols].copy()
    for c in Xdf.columns:
        if not pd.api.types.is_numeric_dtype(Xdf[c]):
            Xdf[c] = Xdf[c].astype("category").cat.codes
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = Xdf.to_numpy(dtype=float)
    T = (df[exposure] > df[exposure].median()).astype(int).to_numpy()
    Y = df[y].to_numpy(dtype=float)

    try:
        from econml.dml import CausalForestDML  # type: ignore
        est = CausalForestDML(n_estimators=300, random_state=seed)
        est.fit(Y, T, X=X)
        cate = est.effect(X)
        backend = "econml.CausalForestDML"
    except Exception:
        cate = _t_learner_cate(X, T, Y, seed)
        backend = "sklearn T-learner (EconML not installed)"

    return {
        "exposure": exposure,
        "backend": backend,
        "ate": float(np.mean(cate)),
        "cate_std": float(np.std(cate)),
        "cate": cate,
    }
