"""
Dataset loading for RFMiD retinal fundus images, plus a synthetic generator.

The synthetic generator produces fundus-like images with bright vessel-like
structures and a matching metadata table. It lets the entire pipeline (and the
test suite) run end-to-end without the real download, which is useful for CI
and for verifying the code before applying for / downloading data.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from skimage.exposure import equalize_adapthist


@dataclass
class Sample:
    """A single example flowing through the pipeline."""
    patient_id: str
    image: np.ndarray            # H x W x 3, uint8
    metadata: dict               # clinical covariates


class RetinalDataset:
    """
    Loads mBRSET images + metadata, or generates synthetic equivalents.

    Parameters
    ----------
    cfg : dict
        The parsed configuration (see configs/default.yaml).
    synthetic : bool
        If True, ignore disk and generate random data.
    n_synthetic : int
        Number of synthetic samples to generate when synthetic=True.
    """

    def __init__(self, cfg: dict, synthetic: bool = False,
                 n_synthetic: int = 64, split: Optional[str] = None):
        self.cfg = cfg
        self.synthetic = synthetic
        self.split = split
        self.image_size = cfg["data"]["image_size"]
        self._rng = np.random.default_rng(cfg.get("seed", 42))

        if synthetic:
            self.metadata = self._make_synthetic_metadata(n_synthetic)
            self._images = None  # generated lazily
        else:
            self._images_dir, self._csv_path = self._resolve_paths()
            self.metadata = self._load_real_metadata()
            self._images = None

    # ------------------------------------------------------------------ #
    # Real data
    # ------------------------------------------------------------------ #
    def _resolve_paths(self) -> "tuple[str, str]":
        """
        Return (images_dir, metadata_csv_path). When `self.split` is set,
        resolves against cfg['data']['splits'][split] (RFMiD's official
        Training/Validation/Testing partitions); otherwise preserves the
        original flat root/images_subdir/metadata_csv layout unchanged.
        """
        d = self.cfg["data"]
        if self.split is not None:
            splits = d.get("splits", {})
            if self.split not in splits:
                raise KeyError(
                    f"Split '{self.split}' not in cfg['data']['splits']; "
                    f"available: {list(splits)}")
            s = splits[self.split]
            root = d.get("root", "")
            return (os.path.join(root, s["images_dir"]),
                    os.path.join(root, s["metadata_csv"]))
        return (os.path.join(d["root"], d["images_subdir"]),
                os.path.join(d["root"], d["metadata_csv"]))

    def _load_real_metadata(self) -> pd.DataFrame:
        csv_path = self._csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Metadata CSV not found at {csv_path}. "
                f"Set cfg['data']['root'] to the RFMiD dataset directory."
            )
        df = pd.read_csv(csv_path)
        return self._normalise_columns(df)

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map RFMiD column names to the canonical names used internally."""
        cols = self.cfg["data"]["columns"]
        rename = {}
        for canonical, actual in cols.items():
            if actual in df.columns:
                rename[actual] = canonical
            else:
                warnings.warn(
                    f"Column '{actual}' (for '{canonical}') not found in "
                    f"metadata; it will be imputed."
                )
        df = df.rename(columns=rename)

        # Ensure required canonical columns exist; impute if missing.
        if "patient_id" not in df.columns:
            df["patient_id"] = [f"p{i:05d}" for i in range(len(df))]
        if "age" not in df.columns:
            df["age"] = self._rng.normal(60, 12, len(df)).clip(18, 95)
        if "sex" not in df.columns:
            df["sex"] = self._rng.choice(["M", "F"], len(df))
        if "diabetes_time" not in df.columns:
            df["diabetes_time"] = self._rng.exponential(6, len(df)).clip(0, 40)
        if "systolic_bp" not in df.columns:
            # Impute BP with mild dependence on age (keeps mediation realistic).
            df["systolic_bp"] = (
                110 + 0.4 * (df["age"] - 50)
                + self._rng.normal(0, 12, len(df))
            ).clip(90, 200)
        if "ldl_cholesterol" not in df.columns:
            df["ldl_cholesterol"] = self._rng.normal(
                120, 30, len(df)
            ).clip(40, 260)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Synthetic data
    # ------------------------------------------------------------------ #
    def _make_synthetic_metadata(self, n: int) -> pd.DataFrame:
        rng = self._rng
        age = rng.normal(60, 12, n).clip(18, 95)
        sex = rng.choice(["M", "F"], n)
        dm_time = rng.exponential(6, n).clip(0, 40)
        # BP and LDL depend on age/diabetes so mediation has real signal.
        sbp = (110 + 0.45 * (age - 50) + 1.2 * dm_time
               + rng.normal(0, 10, n)).clip(90, 200)
        ldl = (115 + 0.8 * dm_time + rng.normal(0, 25, n)).clip(40, 260)
        return pd.DataFrame({
            "patient_id": [f"syn{i:05d}" for i in range(n)],
            "age": age,
            "sex": sex,
            "diabetes_time": dm_time,
            "systolic_bp": sbp,
            "ldl_cholesterol": ldl,
        })

    def _synthetic_image(self, idx: int) -> np.ndarray:
        """
        Generate a fundus-like RGB image: dark reddish background with a few
        bright curved vessel strokes. Vessel density is correlated with the
        patient's age so downstream causal analysis has real structure.
        """
        s = self.image_size
        rng = np.random.default_rng(self.cfg.get("seed", 42) + idx)
        img = np.zeros((s, s, 3), dtype=np.float32)
        # Reddish circular fundus background.
        yy, xx = np.mgrid[0:s, 0:s]
        cx, cy = s / 2, s / 2
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) < (0.46 * s) ** 2
        img[..., 0] = np.where(disk, 90, 0)
        img[..., 1] = np.where(disk, 25, 0)
        img[..., 2] = np.where(disk, 20, 0)

        # More vessels for older patients (drives the causal signal).
        age = float(self.metadata.iloc[idx]["age"])
        n_vessels = int(8 + (age - 40) / 4 + rng.normal(0, 2))
        n_vessels = max(4, min(n_vessels, 30))

        for _ in range(n_vessels):
            t = np.linspace(0, 1, 200)
            x0, y0 = rng.uniform(0.3, 0.7, 2) * s
            ang = rng.uniform(0, 2 * np.pi)
            curve = rng.uniform(-0.4, 0.4)
            xs = x0 + np.cos(ang) * t * 0.45 * s + curve * (t ** 2) * s
            ys = y0 + np.sin(ang) * t * 0.45 * s - curve * (t ** 2) * s
            for x, y in zip(xs, ys):
                xi, yi = int(x), int(y)
                if 1 <= xi < s - 1 and 1 <= yi < s - 1:
                    img[yi - 1:yi + 2, xi - 1:xi + 2, :] += 120
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    # ------------------------------------------------------------------ #
    # Standard dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Sample:
        row = self.metadata.iloc[idx]
        if self.synthetic:
            image = self._synthetic_image(idx)
        else:
            image = self._load_real_image(row)
        return Sample(
            patient_id=str(row["patient_id"]),
            image=image,
            metadata=row.to_dict(),
        )

    @staticmethod
    def _clahe_green(img_uint8: np.ndarray) -> np.ndarray:
        """Apply CLAHE to green channel — must match DRIVE training preprocessing."""
        img = img_uint8.astype(np.float32) / 255.0
        img[:, :, 1] = equalize_adapthist(img[:, :, 1], clip_limit=0.03)
        return (img * 255).astype(np.uint8)

    def _load_real_image(self, row: pd.Series) -> np.ndarray:
        img_dir = self._images_dir
        candidates = [
            row.get("image_id"),
            row.get("file"),
            f"{row['patient_id']}.jpg",
            f"{row['patient_id']}.png",
        ]
        for name in candidates:
            if name is None:
                continue
            path = os.path.join(img_dir, str(name))
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                img = img.resize((self.image_size, self.image_size))
                return self._clahe_green(np.array(img))
        raise FileNotFoundError(
            f"No image found for patient {row['patient_id']} in {img_dir}."
        )

    def metadata_frame(self) -> pd.DataFrame:
        """Return a copy of the clinical metadata table."""
        return self.metadata.copy()
