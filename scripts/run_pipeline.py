#!/usr/bin/env python
"""
Command-line entry point.

Examples
--------
# Dry run on synthetic data (no dataset needed):
python scripts/run_pipeline.py --config configs/default.yaml --synthetic

# Real run once mBRSET is placed under data/mbrset/:
python scripts/run_pipeline.py --config configs/default.yaml
"""

import argparse
import os
import sys

import yaml

# Make `src` importable when run from the repo root.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Retinal CVD causal + UQ pipeline")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--synthetic", action="store_true",
                   help="run on generated synthetic data (no download)")
    p.add_argument("--n-synthetic", type=int, default=64,
                   help="number of synthetic samples")
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    run(cfg, synthetic=args.synthetic, n_synthetic=args.n_synthetic)


if __name__ == "__main__":
    main()
