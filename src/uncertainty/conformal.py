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

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import f1_score, roc_auc_score
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
    is_binary_outcome: bool = False
    clipped_to: Optional[Tuple[float, float]] = None


def split_conformal_regression(X: np.ndarray, y: np.ndarray, cfg: dict,
                               X_holdout: Optional[np.ndarray] = None,
                               y_holdout: Optional[np.ndarray] = None,
                               ) -> ConformalRegressionResult:
    alpha = cfg["uncertainty"]["alpha"]
    cal_frac = cfg["uncertainty"]["calibration_fraction"]
    test_frac = cfg["uncertainty"]["test_fraction"]
    seed = cfg.get("seed", 42)

    # Regressing a binary outcome with an unconstrained regressor produces
    # nonconformity scores that aren't bounded by the outcome's true range,
    # so the resulting interval can (and in practice does) exceed the
    # entire achievable scale. Detect this and clip to the outcome's known
    # support -- clipping to a TRUE physical bound is coverage-safe (a
    # covered point stays covered), unlike clipping to sample min/max.
    binary_outcome = len(np.unique(y)) <= 2
    bounds = cfg["uncertainty"].get("outcome_bounds")
    if bounds is None and binary_outcome:
        bounds = (0.0, 1.0)
    if binary_outcome:
        warnings.warn(
            "split_conformal_regression called on a binary/near-binary "
            "outcome (<=2 unique values); consider split_conformal_classification "
            f"instead. Intervals will be clipped to {bounds}.",
            stacklevel=2)

    if X_holdout is not None:
        # An external holdout replaces the internal test split; only a
        # train/calibration split is needed from X, y.
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y, test_size=cal_frac, random_state=seed)
        X_te, y_te = X_holdout, y_holdout
    else:
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

    clipped_to = None
    if bounds is not None:
        lo_b, hi_b = bounds
        point = np.clip(point, lo_b, hi_b)
        lower = np.clip(lower, lo_b, hi_b)
        upper = np.clip(upper, lo_b, hi_b)
        clipped_to = (float(lo_b), float(hi_b))

    covered = np.mean((y_te >= lower) & (y_te <= upper))

    return ConformalRegressionResult(
        point=point, lower=lower, upper=upper, quantile=q,
        empirical_coverage=float(covered),
        mean_width=float(np.mean(upper - lower)),
        alpha=alpha,
        is_binary_outcome=binary_outcome,
        clipped_to=clipped_to,
    )


@dataclass
class ConformalClassificationResult:
    prediction_sets: list
    empirical_coverage: float
    mean_set_size: float
    alpha: float
    auroc: float = float("nan")
    f1: float = 0.0
    coverage_by_class: Dict[int, float] = None


def split_conformal_classification(X: np.ndarray, y: np.ndarray, cfg: dict,
                                   X_holdout: Optional[np.ndarray] = None,
                                   y_holdout: Optional[np.ndarray] = None,
                                   ) -> ConformalClassificationResult:
    """APS-style conformal sets for the binary CVD label."""
    alpha = cfg["uncertainty"]["alpha"]
    cal_frac = cfg["uncertainty"]["calibration_fraction"]
    test_frac = cfg["uncertainty"]["test_fraction"]
    seed = cfg.get("seed", 42)

    if X_holdout is not None:
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y, test_size=cal_frac, random_state=seed, stratify=y)
        X_te, y_te = X_holdout, y_holdout
    else:
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

    y_te_arr = np.asarray(y_te)
    te_proba = clf.predict_proba(X_te)
    pred_sets, sizes, covered = [], [], []
    for i in range(len(X_te)):
        keep = classes[(1.0 - te_proba[i]) <= qhat]
        pred_sets.append(set(keep.tolist()))
        sizes.append(len(keep))
        covered.append(y_te_arr[i] in keep)

    # Discrimination metrics for the underlying classifier (not just the
    # conformal wrapper's coverage/set-size), using whichever class sklearn
    # treats as positive (the larger label, matching a 0/1 convention).
    pos_label = int(classes.max())
    pos_idx = int(np.where(classes == pos_label)[0][0])
    auroc = (float(roc_auc_score(y_te_arr, te_proba[:, pos_idx]))
             if len(np.unique(y_te_arr)) > 1 else float("nan"))
    hard_pred = classes[np.argmax(te_proba, axis=1)]
    f1 = float(f1_score(y_te_arr, hard_pred, pos_label=pos_label, zero_division=0))

    # Class-conditional (Mondrian) coverage -- the marginal guarantee alone
    # can hide a class-dependent failure under class imbalance.
    covered_arr = np.asarray(covered)
    coverage_by_class = {
        int(c): (float(covered_arr[y_te_arr == c].mean())
                 if np.any(y_te_arr == c) else float("nan"))
        for c in classes
    }

    return ConformalClassificationResult(
        prediction_sets=pred_sets,
        empirical_coverage=float(np.mean(covered)),
        mean_set_size=float(np.mean(sizes)),
        alpha=alpha,
        auroc=auroc,
        f1=f1,
        coverage_by_class=coverage_by_class,
    )
