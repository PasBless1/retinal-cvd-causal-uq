"""
PyTorch Dataset for the DRIVE retinal vessel segmentation benchmark.

DRIVE structure (after extraction):
    training/images/*.tif       -- 20 RGB fundus images
    training/1st_manual/*.gif   -- 20 ground-truth vessel masks
    training/mask/*.gif         -- 20 field-of-view masks

The 20 training images are split into train / val by index.
The test split has no public manual annotations so is used for
inference only (apply the trained model to produce segmentations).

Preprocessing:
  - CLAHE on the green channel (most informative for vessel contrast)
  - Resize to image_size x image_size
  - Normalise to [0, 1]

Augmentation (train split only):
  - Random horizontal / vertical flip
  - Random 90-degree rotation (k in {0,1,2,3})
  - Random brightness / contrast jitter via numpy
"""

from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist
from torch.utils.data import Dataset
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def _clahe_green(img_uint8: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the green channel; return float32 H×W×3 in [0,1]."""
    img = img_uint8.astype(np.float32) / 255.0
    green = img[:, :, 1]
    img[:, :, 1] = equalize_adapthist(green, clip_limit=0.03)
    return img


def _load_gif_mask(path: str, size: int) -> np.ndarray:
    """Load a .gif mask, resize (nearest), return binary float32 H×W."""
    mask = Image.open(path).convert("L")
    mask = mask.resize((size, size), Image.NEAREST)
    return (np.array(mask) > 127).astype(np.float32)


def _load_tif_image(path: str, size: int) -> np.ndarray:
    """Load a .tif fundus image, resize (bilinear), return uint8 H×W×3."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def _augment(img: np.ndarray, mask: np.ndarray,
             fov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random flip + rotation applied identically to image, mask, and FOV."""
    if random.random() > 0.5:           # horizontal flip
        img = img[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
        fov = fov[:, ::-1].copy()
    if random.random() > 0.5:           # vertical flip
        img = img[::-1].copy()
        mask = mask[::-1].copy()
        fov = fov[::-1].copy()
    k = random.randint(0, 3)            # random 90° rotation
    if k:
        img = np.rot90(img, k).copy()
        mask = np.rot90(mask, k).copy()
        fov = np.rot90(fov, k).copy()
    # Brightness / contrast jitter on the float image
    alpha = random.uniform(0.8, 1.2)    # contrast
    beta = random.uniform(-0.08, 0.08)  # brightness
    img = np.clip(alpha * img + beta, 0.0, 1.0)
    return img, mask, fov


# ── dataset ──────────────────────────────────────────────────────────────────

class DRIVEDataset(Dataset):
    """
    Parameters
    ----------
    drive_dir : str
        Path to the extracted DRIVE root (contains training/ and test/).
    split : str
        'train', 'val', or 'test'.  train/val share the 20 labelled training
        images; test uses the 20 unlabelled test images.
    image_size : int
        Resize target (square).
    augment : bool
        Apply random augmentation (recommended for train split only).
    val_ids : list[int] or None
        Zero-based indices from the 20 training images reserved for
        validation.  Defaults to the last 4 images [16,17,18,19].
    """

    def __init__(
        self,
        drive_dir: str,
        split: str = "train",
        image_size: int = 512,
        augment: bool = True,
        val_ids: Optional[List[int]] = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")

        if val_ids is None:
            val_ids = [16, 17, 18, 19]
        val_ids_set = set(val_ids)

        if split in ("train", "val"):
            self._load_labelled(drive_dir, val_ids_set)
        else:
            self._load_test(drive_dir)

    # ------------------------------------------------------------------ #

    def _load_labelled(self, drive_dir: str, val_ids_set: set) -> None:
        img_dir = os.path.join(drive_dir, "training", "images")
        mask_dir = os.path.join(drive_dir, "training", "1st_manual")
        fov_dir = os.path.join(drive_dir, "training", "mask")

        img_files = sorted(
            f for f in os.listdir(img_dir) if f.endswith(".tif")
        )
        mask_files = sorted(
            f for f in os.listdir(mask_dir) if f.endswith(".gif")
        )
        fov_files = sorted(
            f for f in os.listdir(fov_dir) if f.endswith(".gif")
        )

        assert len(img_files) == len(mask_files) == 20, (
            f"Expected 20 training images/masks, found {len(img_files)}/{len(mask_files)}"
        )

        all_indices = list(range(20))
        if self.split == "val":
            indices = [i for i in all_indices if i in val_ids_set]
        else:
            indices = [i for i in all_indices if i not in val_ids_set]

        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.fovs: List[np.ndarray] = []
        self.names: List[str] = []

        for i in indices:
            img = _load_tif_image(
                os.path.join(img_dir, img_files[i]), self.image_size
            )
            self.images.append(_clahe_green(img))
            self.masks.append(
                _load_gif_mask(os.path.join(mask_dir, mask_files[i]),
                               self.image_size)
            )
            fov_path = os.path.join(fov_dir, fov_files[i]) if fov_files else None
            if fov_path and os.path.exists(fov_path):
                self.fovs.append(
                    _load_gif_mask(fov_path, self.image_size)
                )
            else:
                self.fovs.append(np.ones((self.image_size, self.image_size),
                                         dtype=np.float32))
            self.names.append(img_files[i])

    def _load_test(self, drive_dir: str) -> None:
        img_dir = os.path.join(drive_dir, "test", "images")
        fov_dir = os.path.join(drive_dir, "test", "mask")

        img_files = sorted(
            f for f in os.listdir(img_dir) if f.endswith(".tif")
        )
        fov_files = sorted(
            f for f in os.listdir(fov_dir) if f.endswith(".gif")
        ) if os.path.isdir(fov_dir) else []

        self.images = []
        self.masks = []     # empty — no ground truth for test
        self.fovs = []
        self.names = []

        for i, fname in enumerate(img_files):
            img = _load_tif_image(
                os.path.join(img_dir, fname), self.image_size
            )
            self.images.append(_clahe_green(img))
            self.masks.append(None)
            fov_path = (os.path.join(fov_dir, fov_files[i])
                        if i < len(fov_files) else None)
            if fov_path and os.path.exists(fov_path):
                self.fovs.append(
                    _load_gif_mask(fov_path, self.image_size)
                )
            else:
                self.fovs.append(np.ones((self.image_size, self.image_size),
                                         dtype=np.float32))
            self.names.append(fname)

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx].copy()       # float32 H×W×3
        fov = self.fovs[idx].copy()         # float32 H×W

        has_label = (self.split in ("train", "val") and
                     self.masks[idx] is not None)
        mask = self.masks[idx].copy() if has_label else np.zeros_like(fov)

        if self.augment:
            img, mask, fov = _augment(img, mask, fov)

        x = torch.tensor(img).permute(2, 0, 1).float()          # 3×H×W
        y = torch.tensor(mask).unsqueeze(0).float()              # 1×H×W
        f = torch.tensor(fov).unsqueeze(0).float()               # 1×H×W

        return x, y, f
