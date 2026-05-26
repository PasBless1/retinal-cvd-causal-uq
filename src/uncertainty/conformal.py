"""
Split (inductive) conformal prediction for CVD risk.

Given any point predictor f trained on a proper training split, we use a
disjoint calibration split to compute nonconformity scores and a finite-sample
quantile. The resulting intervals satisfy

    P( Y_{n+1} in C(X_{n+1}) ) >= 1 - alpha

marginally, under only the exchangeability assumption -- no distributional or
model-correctness assumptions. Both a regression variant (risk score with
intervals) and a classification variant (prediction sets for the binary CVD
label) are provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ConformalRegressionResult:
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quantile: float
    empirical_coverage: float
    mean_width: float
    alpha: float


def split_conformal_regression(X: np.ndarray, y: np.ndarray,
                               cfg: dict) -> ConformalRegressionResult:
    alpha = cfg["uncertainty"]["alpha"]
    cal_frac = cfg["uncertainty"]["calibration_fraction"]
    test_frac = cfg["uncertainty"]["test_fraction"]
    seed = cfg.get("seed", 42)

    # train / calibration / test split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=cal_frac + test_frac, random_state=seed)
    rel_test = test_frac / (cal_frac + test_frac)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, random_state=seed)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, random_state=seed)),
    ])
    model.fit(X_tr, y_tr)

    # Absolute-residual nonconformity scores on the calibration set.
    cal_scores = np.abs(y_cal - model.predict(X_cal))
    n = len(cal_scores)
    # Finite-sample-valid conformal quantile level.
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(cal_scores, level, method="higher"))

    point = model.predict(X_te)
    lower, upper = point - q, point + q
    covered = np.mean((y_te >= lower) & (y_te <= upper))

    return ConformalRegressionResult(
        point=point, lower=lower, upper=upper, quantile=q,
        empirical_coverage=float(covered),
        mean_width=float(np.mean(upper - lower)),
        alpha=alpha,
    )


@dataclass
class ConformalClassificationResult:
    prediction_sets: list
    empirical_coverage: float
    mean_set_size: float
    alpha: float


def split_conformal_classification(X: np.ndarray, y: np.ndarray,
                                   cfg: dict) -> ConformalClassificationResult:
    """APS-style conformal sets for the binary CVD label."""
    alpha = cfg["uncertainty"]["alpha"]
    cal_frac = cfg["uncertainty"]["calibration_fraction"]
    test_frac = cfg["uncertainty"]["test_fraction"]
    seed = cfg.get("seed", 42)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=cal_frac + test_frac, random_state=seed,
        stratify=y)
    rel_test = test_frac / (cal_frac + test_frac)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, random_state=seed,
        stratify=y_tmp)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=seed)),
    ])
    clf.fit(X_tr, y_tr)
    classes = clf.named_steps["gb"].classes_

    cal_proba = clf.predict_proba(X_cal)
    # Nonconformity = 1 - softmax prob of the true class.
    true_idx = np.array([np.where(classes == v)[0][0] for v in y_cal])
    cal_scores = 1.0 - cal_proba[np.arange(len(y_cal)), true_idx]
    n = len(cal_scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = float(np.quantile(cal_scores, level, method="higher"))

    te_proba = clf.predict_proba(X_te)
    pred_sets, sizes, covered = [], [], []
    for i in range(len(X_te)):
        keep = classes[(1.0 - te_proba[i]) <= qhat]
        pred_sets.append(set(keep.tolist()))
        sizes.append(len(keep))
        covered.append(y_te[i] in keep)

    return ConformalClassificationResult(
        prediction_sets=pred_sets,
        empirical_coverage=float(np.mean(covered)),
        mean_set_size=float(np.mean(sizes)),
        alpha=alpha,
    )
