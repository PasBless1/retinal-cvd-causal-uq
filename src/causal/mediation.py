"""
Causal mediation analysis for retinal feature -> CVD risk pathways.

For each (exposure X, mediator M) pair we fit the standard two-model
decomposition under the sequential-ignorability assumption:

    Mediator model :  M = a0 + a*X + theta'C + e_M
    Outcome model  :  Y = b0 + c'*X + b*M + phi'C + e_Y

with confounders C adjusted for in both models. We report:

    NDE (natural direct effect)   = c'
    NIE (natural indirect effect) = a * b
    Total effect                  = c' + a*b
    Proportion mediated           = NIE / Total

Confidence intervals come from a nonparametric bootstrap, which avoids the
fragile normality assumptions of the product-of-coefficients delta method.

NOTE: causal validity rests on no-unmeasured-confounding of the X->Y, X->M
and M->Y relationships, and correct model specification. These assumptions are
stated explicitly in the paper's limitations section; the code estimates the
quantities, it does not certify the assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class MediationResult:
    exposure: str
    mediator: str
    direct_effect: float
    indirect_effect: float
    total_effect: float
    proportion_mediated: float
    ci: Dict[str, tuple] = field(default_factory=dict)

    def summary(self) -> str:
        lo_n, hi_n = self.ci.get("indirect", (np.nan, np.nan))
        return (
            f"{self.exposure:>22s} -> {self.mediator:<16s} | "
            f"NDE={self.direct_effect:+.4f}  "
            f"NIE={self.indirect_effect:+.4f} "
            f"[{lo_n:+.4f}, {hi_n:+.4f}]  "
            f"prop_med={self.proportion_mediated:5.1%}"
        )


def _design(df: pd.DataFrame, confounders: List[str]) -> str:
    """Build the additive confounder term for a patsy formula."""
    safe = []
    for c in confounders:
        if df[c].dtype == object or str(df[c].dtype).startswith("category"):
            safe.append(f"C({c})")
        else:
            safe.append(c)
    return " + ".join(safe) if safe else "1"


def _single_mediation(df: pd.DataFrame, x: str, m: str, y: str,
                       confounders: List[str]) -> Dict[str, float]:
    c_term = _design(df, confounders)
    med_model = smf.ols(f"{m} ~ {x} + {c_term}", data=df).fit()
    out_model = smf.ols(f"{y} ~ {x} + {m} + {c_term}", data=df).fit()
    a = med_model.params.get(x, 0.0)
    b = out_model.params.get(m, 0.0)
    c_prime = out_model.params.get(x, 0.0)
    nie = a * b
    nde = c_prime
    total = nie + nde
    prop = nie / total if abs(total) > 1e-9 else 0.0
    return {"a": a, "b": b, "nde": nde, "nie": nie,
            "total": total, "prop": prop}


def run_mediation(df: pd.DataFrame, cfg: dict) -> List[MediationResult]:
    """
    Run mediation for every (exposure, mediator) pair in the config and
    attach bootstrap confidence intervals.
    """
    exposures = cfg["causal"]["exposures"]
    mediators = cfg["causal"]["mediators"]
    confounders = cfg["causal"]["confounders"]
    y = cfg["causal"]["outcome"]
    n_boot = cfg["causal"]["bootstrap_iterations"]
    rng = np.random.default_rng(cfg.get("seed", 42))

    results: List[MediationResult] = []
    for x in exposures:
        if x not in df.columns:
            continue
        for m in mediators:
            if m not in df.columns:
                continue
            point = _single_mediation(df, x, m, y, confounders)

            boot_nie, boot_nde, boot_prop = [], [], []
            idx = np.arange(len(df))
            for _ in range(n_boot):
                bi = rng.choice(idx, size=len(df), replace=True)
                bdf = df.iloc[bi]
                try:
                    bp = _single_mediation(bdf, x, m, y, confounders)
                    boot_nie.append(bp["nie"])
                    boot_nde.append(bp["nde"])
                    boot_prop.append(bp["prop"])
                except Exception:
                    continue

            def _ci(arr):
                if not arr:
                    return (np.nan, np.nan)
                return (float(np.percentile(arr, 2.5)),
                        float(np.percentile(arr, 97.5)))

            results.append(MediationResult(
                exposure=x,
                mediator=m,
                direct_effect=point["nde"],
                indirect_effect=point["nie"],
                total_effect=point["total"],
                proportion_mediated=point["prop"],
                ci={
                    "indirect": _ci(boot_nie),
                    "direct": _ci(boot_nde),
                    "proportion": _ci(boot_prop),
                },
            ))
    return results


def results_to_frame(results: List[MediationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "exposure": r.exposure,
            "mediator": r.mediator,
            "direct_effect": r.direct_effect,
            "indirect_effect": r.indirect_effect,
            "total_effect": r.total_effect,
            "proportion_mediated": r.proportion_mediated,
            "indirect_ci_low": r.ci["indirect"][0],
            "indirect_ci_high": r.ci["indirect"][1],
            "proportion_ci_low": r.ci["proportion"][0],
            "proportion_ci_high": r.ci["proportion"][1],
        })
    return pd.DataFrame(rows)
