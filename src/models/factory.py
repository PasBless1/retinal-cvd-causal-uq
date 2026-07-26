"""Model factory + a minimal trainer used by the pipeline."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from .bayesian_unet import BayesianUNet, elbo_loss
from .mc_dropout_unet import MCDropoutUNet


def build_segmenter(cfg: dict) -> torch.nn.Module:
    name = cfg["segmentation"]["model"]
    base = cfg["segmentation"]["base_channels"]
    if name == "bayesian_unet":
        return BayesianUNet(base=base)
    if name == "mc_dropout_unet":
        return MCDropoutUNet(base=base, p=cfg["segmentation"]["dropout_p"])
    raise ValueError(f"Unknown segmentation model: {name}")


def _pseudo_vessel_target(images: torch.Tensor) -> torch.Tensor:
    """
    Build a weak vessel label from image brightness as a stand-in for true
    annotations. With the real dataset, replace this with masks from a
    DRIVE-pretrained network or manual annotations; the training loop is
    unchanged.
    """
    gray = images.mean(dim=1, keepdim=True)
    thr = gray.flatten(2).quantile(0.85, dim=2).view(-1, 1, 1, 1)
    return (gray > thr).float()


def train_segmenter(model: torch.nn.Module, images_np: np.ndarray,
                     cfg: dict) -> torch.nn.Module:
    """
    Train (or load) the segmenter.

    images_np : N x H x W x 3 uint8 array.
    """
    ckpt = cfg["segmentation"]["checkpoint"]
    if os.path.exists(ckpt):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device)
        model.eval()
        return model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    x = torch.tensor(images_np).permute(0, 3, 1, 2).float() / 255.0
    n = x.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["segmentation"]["lr"])
    bs = cfg["segmentation"]["batch_size"]

    model.train()
    for epoch in range(cfg["segmentation"]["epochs"]):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = x[idx].to(device)
            yb = _pseudo_vessel_target(xb).to(device)
            out = model(xb)
            kl = model.kl_divergence().to(device)
            loss = elbo_loss(out["logits"], yb, kl, dataset_size=n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"[segmenter] epoch {epoch + 1}"
              f"/{cfg['segmentation']['epochs']} loss={epoch_loss:.3f}")

    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    model.eval()
    return model


def segment_with_uncertainty(model: torch.nn.Module,
                             images_np: np.ndarray,
                             cfg: dict,
                             batch_size: int = 8) -> Dict[str, np.ndarray]:
    """Run batched MC inference, returning mean masks and uncertainty maps."""
    device = next(model.parameters()).device
    x = torch.tensor(images_np).permute(0, 3, 1, 2).float() / 255.0
    s = cfg["segmentation"]["mc_samples"]
    n = x.shape[0]
    masks, uncs = [], []
    model.eval()
    for start in range(0, n, batch_size):
        batch = x[start:start + batch_size].to(device)
        out = model(batch, mc_samples=s)
        masks.append(out["mask"].squeeze(1).cpu().numpy())      # (B, H, W)
        uncs.append(out["uncertainty"].squeeze(1).cpu().numpy())
        if (start // batch_size) % 10 == 0:
            print(f"  [{start + len(masks[-1])}/{n}]", flush=True)
    return {"masks": np.concatenate(masks), "uncertainty": np.concatenate(uncs)}
