"""
End-to-end pipeline: data -> segmentation+UQ -> features -> causal -> conformal.

Each stage writes artifacts to the output directory so runs are auditable and
resumable. `run(cfg, synthetic)` returns a results dict and is what the CLI
and the test suite call.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from .data.dataset import RetinalDataset
from .data.labels import add_synthetic_cvd_risk
from .models.factory import (build_segmenter, train_segmenter,
                             segment_with_uncertainty)
from .features.extraction import extract_batch
from .causal.mediation import run_mediation, results_to_frame
from .causal.causal_forest import heterogeneous_effects
from .uncertainty.conformal import (split_conformal_regression,
                                    split_conformal_classification)


def _set_seeds(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(cfg: dict, synthetic: bool = False,
        n_synthetic: int = 256) -> Dict[str, object]:
    _set_seeds(cfg.get("seed", 42))
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    save = cfg["output"]["save_intermediate"]

    # ---- Stage 0: data -------------------------------------------------
    print("[stage 0] loading data"
          f" ({'synthetic' if synthetic else 'mBRSET'})")
    ds = RetinalDataset(cfg, synthetic=synthetic, n_synthetic=n_synthetic)
    images = np.stack([ds[i].image for i in range(len(ds))])
    meta = ds.metadata_frame()

    # ---- Stage 1: segmentation + uncertainty ---------------------------
    print("[stage 1] segmentation with uncertainty")
    model = build_segmenter(cfg)
    model = train_segmenter(model, images, cfg)
    seg = segment_with_uncertainty(model, images, cfg)
    if save:
        np.save(os.path.join(out_dir, "uncertainty.npy"),
                seg["uncertainty"])

    # ---- Stage 2: biomarker extraction ---------------------------------
    print("[stage 2] biomarker extraction")
    feats = extract_batch(seg["masks"], seg["uncertainty"], cfg)
    feat_df = pd.DataFrame(feats)
    df = pd.concat([meta.reset_index(drop=True),
                    feat_df.reset_index(drop=True)], axis=1)

    # ---- Stage 2b: labels ---------------------------------------------
    # Synthetic runs have no real disease columns to fall back on, so a
    # synthetic outcome is required regardless of `labels.enabled` (which
    # only governs whether to synthesize on top of real RFMiD data).
    if (synthetic or cfg["labels"]["enabled"]) and "cvd_risk" not in df.columns:
        df = add_synthetic_cvd_risk(df, cfg)
    if save:
        df.to_csv(os.path.join(out_dir, "features_table.csv"), index=False)

    # ---- Stage 3: causal mediation ------------------------------------
    print("[stage 3] causal mediation analysis")
    med_results = run_mediation(df, cfg)
    med_df = results_to_frame(med_results)
    if save:
        med_df.to_csv(os.path.join(out_dir, "mediation_results.csv"),
                      index=False)
    for r in med_results:
        print("   " + r.summary())

    # Heterogeneous effects for the first available exposure.
    het = None
    for x in cfg["causal"]["exposures"]:
        if x in df.columns:
            het = heterogeneous_effects(df, x, cfg)
            print(f"   [causal forest] {het['exposure']}: "
                  f"ATE={het['ate']:+.4f} (sd {het['cate_std']:.4f}) "
                  f"via {het['backend']}")
            break

    # ---- Stage 4: conformal prediction --------------------------------
    print("[stage 4] conformal prediction")
    feature_cols = [
        c for c in feat_df.columns
        if c != "segmentation_uncertainty"
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].to_numpy(dtype=float)

    reg = split_conformal_regression(X, df["cvd_risk"].to_numpy(float), cfg)
    reg_tag = " (diagnostic only -- binary outcome, see classification)" if reg.is_binary_outcome else ""
    print(f"   regression{reg_tag}: target coverage "
          f"{1 - reg.alpha:.0%}, empirical "
          f"{reg.empirical_coverage:.1%}, mean width {reg.mean_width:.2f}")

    clf = None
    if "cvd_label" in df.columns and df["cvd_label"].nunique() > 1:
        vc = df["cvd_label"].value_counts()
        if vc.min() >= 10:  # need enough per class for a stratified split
            clf = split_conformal_classification(
                X, df["cvd_label"].to_numpy(int), cfg)
            print(f"   classification: empirical coverage "
                  f"{clf.empirical_coverage:.1%}, mean set size "
                  f"{clf.mean_set_size:.2f}, AUROC {clf.auroc:.3f}, "
                  f"F1 {clf.f1:.3f}")
            by_class = {k: round(v, 3) for k, v in clf.coverage_by_class.items()}
            print(f"   classification: coverage by class {by_class}")
        else:
            print("   classification: skipped (insufficient class balance)")

    # ---- Summary -------------------------------------------------------
    summary = {
        "n_samples": len(df),
        "synthetic": synthetic,
        "mediation": med_df.to_dict(orient="records"),
        "causal_forest": (
            None if het is None
            else {k: v for k, v in het.items() if k != "cate"}
        ),
        "conformal_regression": {
            "target_coverage": 1 - reg.alpha,
            "empirical_coverage": reg.empirical_coverage,
            "mean_interval_width": reg.mean_width,
            "quantile": reg.quantile,
            "is_binary_outcome": reg.is_binary_outcome,
            "clipped_to": reg.clipped_to,
            "note": ("diagnostic only (binary outcome); see "
                     "conformal_classification for the headline UQ result"
                     if reg.is_binary_outcome else None),
        },
        "conformal_classification": (
            None if clf is None else {
                "empirical_coverage": clf.empirical_coverage,
                "mean_set_size": clf.mean_set_size,
                "auroc": clf.auroc,
                "f1": clf.f1,
                "coverage_by_class": clf.coverage_by_class,
            }
        ),
        "headline_uq": "classification" if (clf is not None and reg.is_binary_outcome) else "regression",
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[done] summary written to {out_dir}/summary.json")
    return summary
