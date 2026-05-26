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
from src.data.labels import add_synthetic_cvd_risk
from src.features.extraction import extract_biomarkers
from src.causal.mediation import run_mediation
from src.uncertainty.conformal import split_conformal_regression
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
