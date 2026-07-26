"""
Synthetic CVD risk label construction.

When the dataset has no recorded cardiovascular outcome (mBRSET does not), we
construct a continuous risk score with Framingham-style weighting that also
depends on the extracted retinal biomarkers. This deliberately injects a real
(known) causal structure: retinal density and tortuosity influence risk both
directly and through blood pressure / LDL, so the mediation stage has a ground
truth to recover.

This is a methodological scaffold, not a clinical instrument. With a dataset
that has true outcomes (e.g. UK Biobank MACE codes), delete this module and
point `causal.outcome` at the real column.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# Columns that are never disease indicators, even though some are numeric
# and binary-valued (e.g. sex once encoded) -- excluded from the
# auto-inferred disease-column search in add_noncircular_outcome.
_NON_DISEASE_COLUMNS = {
    "patient_id", "ID", "Disease_Risk", "age", "sex", "diabetes_time",
    "systolic_bp", "ldl_cholesterol", "cvd_risk", "cvd_label",
}


def add_noncircular_outcome(df: pd.DataFrame, cfg: dict,
                            exclude: Optional[List[str]] = None,
                            disease_columns: Optional[List[str]] = None,
                            out_col: str = "Disease_Risk_ex_mediator",
                            ) -> pd.DataFrame:
    """
    Reconstruct RFMiD's composite disease-risk outcome with one or more
    mediator columns excluded from the OR.

    RFMiD's `Disease_Risk` label is 1 if ANY of 46 conditions (including
    DR and BRVO) is positive. Using DR/BRVO as "mediators" of an effect on
    `Disease_Risk` risks tautology: DR being positive can directly cause
    `Disease_Risk` to be positive by definition, independent of any
    biological pathway. This builds an alternative outcome -- the OR of
    every OTHER disease column -- so a mediator can be tested against an
    outcome that does not itself contain that mediator as one of its terms.

    Disease columns are auto-inferred (binary-valued numeric columns not in
    a small blacklist of known non-disease fields) rather than hardcoded,
    since exact RFMiD column spellings can vary across dataset copies.
    """
    exclude = exclude if exclude is not None else list(cfg["causal"]["mediators"])
    if disease_columns is None:
        blacklist = _NON_DISEASE_COLUMNS | {out_col} | set(exclude)
        disease_columns = [
            c for c in df.columns
            if c not in blacklist
            and pd.api.types.is_numeric_dtype(df[c])
            and set(pd.unique(df[c].dropna())) <= {0, 1}
        ]
        if not disease_columns:
            raise ValueError(
                f"No binary disease columns found to build '{out_col}' "
                f"(excluding {exclude}).")
    else:
        disease_columns = [c for c in disease_columns if c not in exclude]

    df = df.copy()
    df[out_col] = (df[disease_columns].fillna(0).astype(int).sum(axis=1) > 0).astype(float)
    return df


def add_synthetic_cvd_risk(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.get("seed", 42))
    df = df.copy()

    age = df["age"].to_numpy(float)
    sex_male = (df["sex"].astype(str).str.upper().str[0] == "M").astype(float)
    dm = df.get("diabetes_time", pd.Series(np.zeros(len(df)))).to_numpy(float)
    sbp = df.get("systolic_bp",
                 pd.Series(np.full(len(df), 120.0))).to_numpy(float)
    ldl = df.get("ldl_cholesterol",
                 pd.Series(np.full(len(df), 120.0))).to_numpy(float)

    # Retinal contribution (direct effect on risk).
    vd = df.get("vessel_density",
                pd.Series(np.zeros(len(df)))).to_numpy(float)
    tort = df.get("tortuosity",
                  pd.Series(np.zeros(len(df)))).to_numpy(float)

    # Normalise retinal features by their std so coefficients are scale-free.
    vd_std = vd.std() + 1e-9
    tort_std = tort.std() + 1e-9

    # Linear predictor (logit-scale-ish), then mapped to 0-100.
    # Retinal coefficients are applied to z-scored features so the effect size
    # is independent of the raw scale of skeleton-density values (~0.005-0.02).
    lp = (
        0.045 * (age - 50)
        + 0.55 * sex_male
        + 0.06 * dm
        + 0.025 * (sbp - 120)
        + 0.012 * (ldl - 120)
        - 1.5 * ((vd - vd.mean()) / vd_std)    # lower density -> higher risk
        + 0.8 * ((tort - tort.mean()) / tort_std)  # more tortuous -> higher risk
        + rng.normal(0, 0.3, len(df))
    )
    risk = 100.0 / (1.0 + np.exp(-lp))
    df["cvd_risk"] = risk

    thr = cfg["labels"]["binary_threshold"]
    # If the fixed threshold yields a near-degenerate class balance (common
    # with the logistic mapping), fall back to the sample median so the
    # downstream classification + conformal stages are well-posed.
    label = (df["cvd_risk"] > thr).astype(int)
    if label.mean() < 0.1 or label.mean() > 0.9:
        thr = float(df["cvd_risk"].median())
        label = (df["cvd_risk"] > thr).astype(int)
    df["cvd_label"] = label
    return df
