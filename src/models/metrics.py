"""
Segmentation quality metrics: Dice, IoU, AUROC, F1, sensitivity, specificity.

Promoted from the ad hoc `evaluate()` function that used to live only inline
in the training notebook (AUROC/F1/sens/spec on DRIVE validation images, no
Dice/IoU) into reusable, independently-testable code. Used both for
reporting segmentation quality and for threshold calibration
(see `calibration.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class SegmentationMetrics:
    dice: float
    iou: float
    auroc: float
    f1: float
    sensitivity: float
    specificity: float
    n_pixels: int


def _flatten_masked(prob: np.ndarray, gt: np.ndarray,
                     fov: Optional[np.ndarray]) -> "tuple[np.ndarray, np.ndarray]":
    prob = np.asarray(prob).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    if fov is not None:
        inside = np.asarray(fov).reshape(-1).astype(bool)
        prob, gt = prob[inside], gt[inside]
    return prob, gt


def _score(prob: np.ndarray, gt: np.ndarray, threshold: float) -> SegmentationMetrics:
    gt_bin = (gt > 0.5).astype(int)
    pred = (prob > threshold).astype(int)

    tp = int(((pred == 1) & (gt_bin == 1)).sum())
    fn = int(((pred == 0) & (gt_bin == 1)).sum())
    tn = int(((pred == 0) & (gt_bin == 0)).sum())
    fp = int(((pred == 1) & (gt_bin == 0)).sum())

    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = float(f1_score(gt_bin, pred, zero_division=0))
    auroc = (float(roc_auc_score(gt_bin, prob))
             if len(np.unique(gt_bin)) > 1 else float("nan"))

    return SegmentationMetrics(
        dice=float(dice), iou=float(iou), auroc=auroc, f1=f1,
        sensitivity=float(sensitivity), specificity=float(specificity),
        n_pixels=int(gt_bin.size),
    )


def segmentation_metrics(prob: np.ndarray, gt: np.ndarray,
                          fov: Optional[np.ndarray] = None,
                          threshold: float = 0.5) -> SegmentationMetrics:
    """Score a single probability map against its ground-truth mask."""
    p, g = _flatten_masked(prob, gt, fov)
    return _score(p, g, threshold)


def segmentation_metrics_dataset(
        probs: Sequence[np.ndarray], gts: Sequence[np.ndarray],
        fovs: Optional[Sequence[np.ndarray]] = None,
        threshold: float = 0.5) -> SegmentationMetrics:
    """Pool pixels across every image before scoring (matches the
    notebook's existing DRIVE-validation convention of computing metrics
    over the concatenated pixel set rather than averaging per-image
    scores)."""
    all_p, all_g = [], []
    for i in range(len(probs)):
        fov = fovs[i] if fovs is not None else None
        p, g = _flatten_masked(probs[i], gts[i], fov)
        all_p.append(p)
        all_g.append(g)
    return _score(np.concatenate(all_p), np.concatenate(all_g), threshold)
