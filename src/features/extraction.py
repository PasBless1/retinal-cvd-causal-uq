"""
Extraction of vascular biomarkers from binary vessel masks.

Ten interpretable features are computed per image. These become the exposures
in the downstream causal mediation analysis, so each is documented with its
clinical rationale.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def _box_count_fractal(skel: np.ndarray, sizes) -> float:
    """Box-counting fractal dimension of the vessel skeleton.

    Higher values indicate a more space-filling, complex vascular tree;
    rarefaction (microvascular dropout) lowers this and is associated with
    cardiovascular risk.
    """
    counts = []
    for s in sizes:
        if s >= min(skel.shape):
            continue
        # Number of s x s boxes containing at least one vessel pixel.
        reduced = skel[:skel.shape[0] // s * s,
                       :skel.shape[1] // s * s]
        reduced = reduced.reshape(reduced.shape[0] // s, s,
                                  reduced.shape[1] // s, s)
        box = reduced.any(axis=(1, 3))
        counts.append(max(box.sum(), 1))
    sizes_used = [s for s in sizes if s < min(skel.shape)]
    if len(counts) < 2:
        return 0.0
    coeffs = np.polyfit(np.log(1.0 / np.array(sizes_used, dtype=float)),
                        np.log(counts), 1)
    return float(coeffs[0])


def _tortuosity(skel: np.ndarray) -> float:
    """Mean curvature proxy: skeleton path length over endpoint distance.

    Increased arteriolar tortuosity is a recognised retinal marker of
    hypertensive microvascular change.
    """
    labeled, n = ndimage.label(skel)
    if n == 0:
        return 0.0
    ratios = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labeled == lab)
        if len(xs) < 5:
            continue
        path_len = len(xs)
        chord = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) + 1e-6
        ratios.append(path_len / chord)
    return float(np.mean(ratios)) if ratios else 0.0


def _count_bifurcations(skel: np.ndarray) -> int:
    """Count skeleton pixels with >=3 neighbours (branch points)."""
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbours = ndimage.convolve(skel.astype(int), k, mode="constant")
    return int(np.sum((skel == 1) & (neighbours >= 3)))


def _mean_width(mask: np.ndarray, skel: np.ndarray) -> float:
    """Average vessel calibre via distance transform on the skeleton."""
    dist = ndimage.distance_transform_edt(mask)
    vals = dist[skel == 1]
    return float(2.0 * vals.mean()) if vals.size else 0.0


def extract_biomarkers(prob_mask: np.ndarray,
                        uncertainty: np.ndarray,
                        cfg: dict) -> Dict[str, float]:
    """
    Compute the biomarker vector for a single image.

    Parameters
    ----------
    prob_mask : float array in [0, 1] from the segmenter.
    uncertainty : same shape, segmentation std-dev (propagated downstream).
    """
    thr = cfg["features"]["vessel_threshold"]
    mask = (prob_mask > thr).astype(np.uint8)
    if mask.sum() == 0:
        mask[mask.shape[0] // 2, mask.shape[1] // 2] = 1  # avoid empty
    skel = skeletonize(mask).astype(np.uint8)
    labeled, n_comp = ndimage.label(skel)

    feats = {
        # Density of the vascular network (microvascular rarefaction marker).
        "vessel_density": float(skel.sum() / skel.size),
        # Fraction of retina covered by vessels (caliber-sensitive).
        "vessel_area": float(mask.sum() / mask.size),
        # Geometric complexity of the tree.
        "fractal_dimension": _box_count_fractal(
            skel, cfg["features"]["fractal_box_sizes"]),
        # Hypertensive tortuosity marker.
        "tortuosity": _tortuosity(skel),
        # Branching richness.
        "num_bifurcations": float(_count_bifurcations(skel)),
        # Mean arteriolar/venular calibre.
        "vessel_width_mean": _mean_width(mask, skel),
        # Network fragmentation (more components = more dropout).
        "num_components": float(n_comp),
        # Longest connected vessel (continuity of major arcades).
        "longest_vessel": float(
            max((np.sum(labeled == i) for i in range(1, n_comp + 1)),
                default=0.0)),
        # Branch density normalised by network size.
        "branching_density": float(
            _count_bifurcations(skel) / (skel.sum() + 1e-6)),
        # Mean segmentation uncertainty (propagated as a covariate / QC flag).
        "segmentation_uncertainty": float(uncertainty.mean()),
    }
    return feats


def extract_batch(masks: np.ndarray, uncertainty: np.ndarray,
                  cfg: dict) -> "list[dict]":
    """Vectorised wrapper over a batch of masks."""
    return [
        extract_biomarkers(masks[i], uncertainty[i], cfg)
        for i in range(masks.shape[0])
    ]
