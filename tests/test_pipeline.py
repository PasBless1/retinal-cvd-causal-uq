"""
Test suite. Everything runs on synthetic data so no download is needed.

Run: pytest -q
"""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import RetinalDataset
from src.data.labels import add_synthetic_cvd_risk, add_noncircular_outcome
from src.features.extraction import extract_biomarkers
from src.causal.mediation import run_mediation, run_mediation_noncircular
from src.uncertainty.conformal import (split_conformal_regression,
                                       split_conformal_classification)
from src.models.metrics import segmentation_metrics, segmentation_metrics_dataset
from src.models.calibration import calibrate_vessel_threshold
from src.pipeline import run


@pytest.fixture(scope="module")
def cfg():
    cfg_path = os.path.join(os.path.dirname(__file__),
                            "..", "configs", "default.yaml")
    with open(cfg_path) as fh:
        c = yaml.safe_load(fh)
    # Make the suite fast.
    c["segmentation"]["epochs"] = 1
    c["segmentation"]["mc_samples"] = 3
    c["segmentation"]["base_channels"] = 8
    c["data"]["image_size"] = 64
    c["segmentation"]["checkpoint"] = "outputs/_test_ckpt.pt"
    c["causal"]["bootstrap_iterations"] = 50
    return c


def test_synthetic_dataset_shapes(cfg):
    ds = RetinalDataset(cfg, synthetic=True, n_synthetic=8)
    assert len(ds) == 8
    sample = ds[0]
    s = cfg["data"]["image_size"]
    assert sample.image.shape == (s, s, 3)
    assert "age" in sample.metadata


def test_biomarker_extraction_keys(cfg):
    rng = np.random.default_rng(0)
    prob = rng.random((64, 64)).astype(np.float32)
    unc = rng.random((64, 64)).astype(np.float32) * 0.1
    feats = extract_biomarkers(prob, unc, cfg)
    for key in ["vessel_density", "fractal_dimension", "tortuosity",
                "num_bifurcations", "segmentation_uncertainty"]:
        assert key in feats
        assert np.isfinite(feats[key])


def test_mediation_runs_and_decomposes(cfg):
    # Build a table where the mediated path is real by construction.
    rng = np.random.default_rng(1)
    n = 200
    import pandas as pd
    x = rng.normal(0, 1, n)
    sbp = 2.0 * x + rng.normal(0, 1, n)        # X -> M
    y = 1.5 * sbp + 0.5 * x + rng.normal(0, 1, n)  # M -> Y, plus direct
    df = pd.DataFrame({
        "vessel_density": x, "systolic_bp": sbp, "cvd_risk": y,
        "age": rng.normal(60, 10, n), "sex": rng.choice(["M", "F"], n),
        "diabetes_time": rng.exponential(5, n),
    })
    cfg2 = dict(cfg)
    cfg2["causal"] = dict(cfg["causal"])
    cfg2["causal"]["exposures"] = ["vessel_density"]
    cfg2["causal"]["mediators"] = ["systolic_bp"]
    res = run_mediation(df, cfg2)
    assert len(res) == 1
    r = res[0]
    # Indirect effect should be clearly positive and dominate.
    assert r.indirect_effect > 0
    assert r.proportion_mediated > 0.5


def test_conformal_coverage(cfg):
    rng = np.random.default_rng(2)
    n = 400
    X = rng.normal(0, 1, (n, 5))
    y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.2]) + rng.normal(0, 0.5, n)
    res = split_conformal_regression(X, y, cfg)
    # Empirical coverage should be near (and not far below) the target.
    assert res.empirical_coverage >= (1 - cfg["uncertainty"]["alpha"]) - 0.12


def test_full_pipeline_synthetic(cfg, tmp_path):
    cfg["output"]["dir"] = str(tmp_path)
    cfg["segmentation"]["checkpoint"] = str(tmp_path / "ckpt.pt")
    summary = run(cfg, synthetic=True, n_synthetic=24)
    assert summary["n_samples"] == 24
    assert "conformal_regression" in summary
    assert os.path.exists(tmp_path / "summary.json")


# ---------------------------------------------------------------------------
# src/models/metrics.py
# ---------------------------------------------------------------------------

def test_segmentation_metrics_perfect_prediction():
    gt = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=float)
    prob = gt.copy()
    m = segmentation_metrics(prob, gt, threshold=0.5)
    assert m.dice == pytest.approx(1.0)
    assert m.iou == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.sensitivity == pytest.approx(1.0)
    assert m.specificity == pytest.approx(1.0)
    assert m.auroc == pytest.approx(1.0)


def test_segmentation_metrics_respects_fov():
    gt = np.array([[0, 1], [1, 0]], dtype=float)
    prob = np.array([[1, 1], [1, 0]], dtype=float)  # wrong at (0, 0)
    fov = np.array([[0, 1], [1, 1]], dtype=float)   # exclude (0, 0)
    m = segmentation_metrics(prob, gt, fov=fov, threshold=0.5)
    # With (0, 0) excluded, the remaining 3 pixels all agree perfectly.
    assert m.dice == pytest.approx(1.0)
    assert m.n_pixels == 3


def test_segmentation_metrics_dataset_pools_pixels():
    gt1, prob1 = np.array([[1, 0]], dtype=float), np.array([[1, 0]], dtype=float)
    gt2, prob2 = np.array([[1, 1]], dtype=float), np.array([[0, 1]], dtype=float)
    pooled = segmentation_metrics_dataset([prob1, prob2], [gt1, gt2], threshold=0.5)
    # Pooled: gt=[1,0,1,1], pred=[1,0,0,1] -> tp=2, tn=1, fn=1, fp=0.
    expected_dice = (2 * 2) / (2 * 2 + 0 + 1)
    assert pooled.dice == pytest.approx(expected_dice)


# ---------------------------------------------------------------------------
# src/models/calibration.py
# ---------------------------------------------------------------------------

def test_calibrate_vessel_threshold_picks_best_dice():
    rng = np.random.default_rng(10)
    gts, probs = [], []
    for _ in range(6):
        gt = (rng.random((20, 20)) > 0.7).astype(float)
        prob = np.where(gt == 1, 0.8, 0.2) + rng.normal(0, 0.02, (20, 20))
        gts.append(gt)
        probs.append(prob)
    result = calibrate_vessel_threshold(probs, gts, metric="dice", n_grid=30)
    # With tightly-separated clusters (~0.2 vs ~0.8), Dice is near-optimal
    # across a wide range of thresholds in between -- assert the achieved
    # score is high rather than pinning an exact tie-broken threshold.
    assert 0.1 < result.best_threshold < 0.9
    assert result.best_score > 0.9


def test_calibrate_vessel_threshold_uses_quantile_grid_when_probs_capped():
    # Mirrors the real observed bug: probabilities never exceed ~0.45.
    rng = np.random.default_rng(11)
    gts, probs = [], []
    for _ in range(6):
        gt = (rng.random((20, 20)) > 0.7).astype(float)
        prob = np.where(gt == 1, 0.4, 0.1) + rng.normal(0, 0.01, (20, 20))
        prob = np.clip(prob, 0.0, 0.45)
        gts.append(gt)
        probs.append(prob)
    result = calibrate_vessel_threshold(probs, gts, metric="dice", n_grid=30)
    assert result.best_threshold < 0.45
    assert result.best_score > 0.8


# ---------------------------------------------------------------------------
# src/uncertainty/conformal.py
# ---------------------------------------------------------------------------

def test_conformal_classification_reports_auroc_f1_and_class_conditional_coverage(cfg):
    rng = np.random.default_rng(20)
    n = 400
    X = rng.normal(0, 1, (n, 4))
    logit = X[:, 0] * 2.0 - X[:, 1] * 1.0
    y = (logit + rng.normal(0, 0.5, n) > 0).astype(int)
    res = split_conformal_classification(X, y, cfg)
    assert res.auroc > 0.6
    assert 0.0 <= res.f1 <= 1.0
    assert set(res.coverage_by_class.keys()) == {0, 1}
    for v in res.coverage_by_class.values():
        assert 0.0 <= v <= 1.0


def test_conformal_regression_binary_outcome_clips_and_warns(cfg):
    rng = np.random.default_rng(21)
    n = 300
    X = rng.normal(0, 1, (n, 3))
    y = (X[:, 0] > 0).astype(float)
    with pytest.warns(UserWarning):
        res = split_conformal_regression(X, y, cfg)
    assert res.is_binary_outcome is True
    assert res.clipped_to == (0.0, 1.0)
    assert np.all(res.lower >= 0.0)
    assert np.all(res.upper <= 1.0)


def test_conformal_regression_continuous_outcome_not_clipped(cfg):
    rng = np.random.default_rng(2)
    n = 400
    X = rng.normal(0, 1, (n, 5))
    y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.2]) + rng.normal(0, 0.5, n)
    res = split_conformal_regression(X, y, cfg)
    assert res.is_binary_outcome is False
    assert res.clipped_to is None


def test_conformal_regression_external_holdout(cfg):
    rng = np.random.default_rng(22)
    coeffs = np.array([1.0, 0.5, -0.3, 0.2])
    X_pool = rng.normal(0, 1, (400, 4))
    y_pool = X_pool @ coeffs + rng.normal(0, 0.4, 400)
    X_hold = rng.normal(0, 1, (100, 4))
    y_hold = X_hold @ coeffs + rng.normal(0, 0.4, 100)
    res = split_conformal_regression(X_pool, y_pool, cfg,
                                     X_holdout=X_hold, y_holdout=y_hold)
    assert len(res.point) == 100
    assert res.empirical_coverage >= 0.5


def test_conformal_classification_external_holdout(cfg):
    rng = np.random.default_rng(23)

    def _make(n):
        X = rng.normal(0, 1, (n, 4))
        y = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
        return X, y

    X_pool, y_pool = _make(400)
    X_hold, y_hold = _make(100)
    res = split_conformal_classification(X_pool, y_pool, cfg,
                                         X_holdout=X_hold, y_holdout=y_hold)
    assert len(res.prediction_sets) == 100
    assert 0.0 <= res.empirical_coverage <= 1.0


# ---------------------------------------------------------------------------
# src/data/dataset.py -- official-split support
# ---------------------------------------------------------------------------

def test_retinal_dataset_split_resolves_paths(cfg, tmp_path):
    import pandas as pd
    from PIL import Image

    root = tmp_path / "rfmid"
    img_dir = root / "images_train"
    img_dir.mkdir(parents=True)
    rng = np.random.default_rng(30)
    arr = (rng.random((20, 20, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(img_dir / "1.png")

    pd.DataFrame({"ID": [1], "age": [55]}).to_csv(
        root / "train_labels.csv", index=False)

    cfg2 = dict(cfg)
    cfg2["data"] = dict(cfg["data"])
    cfg2["data"]["root"] = str(root)
    cfg2["data"]["image_size"] = 64
    cfg2["data"]["splits"] = {
        "training": {"images_dir": "images_train",
                     "metadata_csv": "train_labels.csv"},
    }
    ds = RetinalDataset(cfg2, synthetic=False, split="training")
    assert len(ds) == 1
    sample = ds[0]
    assert sample.image.shape == (64, 64, 3)


def test_retinal_dataset_unknown_split_raises(cfg, tmp_path):
    cfg2 = dict(cfg)
    cfg2["data"] = dict(cfg["data"])
    cfg2["data"]["root"] = str(tmp_path)
    cfg2["data"]["splits"] = {
        "training": {"images_dir": "x", "metadata_csv": "y.csv"},
    }
    with pytest.raises(KeyError):
        RetinalDataset(cfg2, synthetic=False, split="bogus")


def test_retinal_dataset_legacy_flat_layout_unchanged(cfg, tmp_path):
    import pandas as pd
    root = tmp_path / "flat"
    root.mkdir()
    pd.DataFrame({"ID": [1, 2], "age": [50, 60]}).to_csv(
        root / "metadata.csv", index=False)

    cfg2 = dict(cfg)
    cfg2["data"] = dict(cfg["data"])
    cfg2["data"]["root"] = str(root)
    cfg2["data"]["images_subdir"] = "."
    cfg2["data"]["metadata_csv"] = "metadata.csv"
    ds = RetinalDataset(cfg2, synthetic=False, split=None)
    assert len(ds) == 2


# ---------------------------------------------------------------------------
# src/data/labels.py -- non-circular outcome
# ---------------------------------------------------------------------------

def test_add_noncircular_outcome_excludes_mediator(cfg):
    import pandas as pd
    df = pd.DataFrame({
        "DR": [1, 0, 0, 0],
        "BRVO": [0, 0, 1, 0],
        "Disease_Risk": [1, 0, 1, 0],
        "age": [50, 60, 55, 45],
    })
    out = add_noncircular_outcome(df, cfg, exclude=["DR"],
                                  out_col="Disease_Risk_ex_DR")
    # Row 0 is Disease_Risk=1 only because DR=1 -- excluding DR must zero it.
    assert out["Disease_Risk_ex_DR"].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_add_noncircular_outcome_auto_infers_disease_columns(cfg):
    import pandas as pd
    df = pd.DataFrame({
        "DR": [0, 1, 0],
        "ARMD": [1, 0, 0],
        "cvd_label": [1, 1, 1],   # blacklisted -- not a disease column
        "age": [50, 60, 70],      # non-binary, excluded regardless
    })
    out = add_noncircular_outcome(df, cfg, exclude=["DR"], out_col="out")
    # Only ARMD should drive "out" (DR excluded; cvd_label/age excluded).
    assert out["out"].tolist() == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# src/causal/mediation.py -- non-circular mediation robustness check
# ---------------------------------------------------------------------------

def test_run_mediation_noncircular_attenuates_tautological_effect(cfg):
    import pandas as pd
    rng = np.random.default_rng(40)
    n = 500
    x = rng.normal(0, 1, n)
    dr = ((2.0 * x + rng.normal(0, 1, n)) > 0).astype(int)         # X -> DR
    other = rng.choice([0, 1], n, p=[0.7, 0.3])                    # independent
    disease_risk = ((dr == 1) | (other == 1)).astype(float)        # circular OR

    df = pd.DataFrame({
        "vessel_density": x,
        "DR": dr,
        "other_condition": other,
        "Disease_Risk": disease_risk,
        "age": rng.normal(60, 10, n),
        "sex": rng.choice(["M", "F"], n),
        "diabetes_time": rng.exponential(5, n),
    })

    cfg2 = dict(cfg)
    cfg2["causal"] = dict(cfg["causal"])
    cfg2["causal"]["exposures"] = ["vessel_density"]
    cfg2["causal"]["mediators"] = ["DR"]
    cfg2["causal"]["outcome"] = "Disease_Risk"
    cfg2["causal"]["bootstrap_iterations"] = 20

    circular = run_mediation(df, cfg2)[0]
    assert abs(circular.indirect_effect) > 0.1

    noncircular_df = run_mediation_noncircular(df, cfg2)
    row = noncircular_df.iloc[0]
    # Once DR's own condition is removed from the outcome (leaving only
    # `other_condition`, independent of both X and DR by construction),
    # the indirect effect should collapse substantially.
    assert abs(row["indirect_effect"]) < abs(circular.indirect_effect) * 0.5
