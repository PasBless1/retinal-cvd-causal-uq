"""
Vessel-probability threshold calibration against DRIVE validation masks.

The committed default (`features.vessel_threshold: 0.5`) was never actually
calibrated against anything -- it silently assumed the segmenter's output
spans a normal [0, 1] probability range. On real RFMiD images the
DRIVE-pretrained segmenter's probability map has been observed to never
exceed ~0.499, which makes a fixed threshold of 0.5 zero out every image's
mask. This module picks a threshold from the *achievable* range of the
model's own output (via quantiles of `probs`, not a fixed linspace), scored
against DRIVE's real expert-annotated validation masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import segmentation_metrics_dataset

_METRIC_FIELDS = ("dice", "iou", "f1", "sensitivity", "specificity", "auroc")


@dataclass
class ThresholdCalibrationResult:
    best_threshold: float
    best_score: float
    metric: str
    curve: pd.DataFrame  # columns: threshold, dice, iou, f1, sensitivity, specificity, auroc


def calibrate_vessel_threshold(
        probs: Sequence[np.ndarray], gts: Sequence[np.ndarray],
        fovs: Optional[Sequence[np.ndarray]] = None,
        thresholds: Optional[Sequence[float]] = None,
        metric: str = "dice",
        n_grid: int = 50) -> ThresholdCalibrationResult:
    if metric not in _METRIC_FIELDS:
        raise ValueError(f"metric must be one of {_METRIC_FIELDS}, got {metric!r}")

    if thresholds is None:
        all_probs = np.concatenate([np.asarray(p).reshape(-1) for p in probs])
        quantiles = np.linspace(0.01, 0.99, n_grid)
        thresholds = np.unique(np.quantile(all_probs, quantiles))

    rows = []
    for thr in thresholds:
        m = segmentation_metrics_dataset(probs, gts, fovs, threshold=float(thr))
        rows.append({"threshold": float(thr), **{f: getattr(m, f) for f in _METRIC_FIELDS}})

    curve = pd.DataFrame(rows)
    scored = curve.dropna(subset=[metric])
    if scored.empty:
        raise ValueError(f"No valid '{metric}' scores across the threshold grid.")
    best_row = scored.loc[scored[metric].idxmax()]

    return ThresholdCalibrationResult(
        best_threshold=float(best_row["threshold"]),
        best_score=float(best_row[metric]),
        metric=metric,
        curve=curve,
    )
