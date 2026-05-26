"""
MC-Dropout U-Net: a cheaper alternative to the fully Bayesian U-Net.

Dropout is kept active at inference and multiple stochastic forward passes
approximate the predictive distribution (Gal & Ghahramani, 2016). Use this
when the Bayesian U-Net is too slow to train; the rest of the pipeline is
agnostic to which segmenter produced the mask + uncertainty.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p),
        )

    def forward(self, x):
        return self.net(x)


class MCDropoutUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1,
                 base: int = 32, p: float = 0.3):
        super().__init__()
        self.enc1 = _DoubleConv(in_ch, base, p)
        self.enc2 = _DoubleConv(base, base * 2, p)
        self.enc3 = _DoubleConv(base * 2, base * 4, p)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base * 2, p)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base, p)
        self.head = nn.Conv2d(base, out_ch, 1)

    def _forward_once(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)

    @staticmethod
    def _enable_dropout(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def forward(self, x: torch.Tensor,
                mc_samples: int = 1) -> Dict[str, torch.Tensor]:
        if self.training:
            logits = self._forward_once(x)
            return {"logits": logits, "mask": torch.sigmoid(logits)}

        self.eval()
        self._enable_dropout(self)  # keep dropout stochastic at test time
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                preds.append(torch.sigmoid(self._forward_once(x)))
        stacked = torch.stack(preds, 0)
        return {
            "mask": stacked.mean(0),
            "uncertainty": stacked.std(0),
            "samples": stacked,
        }

    def kl_divergence(self) -> torch.Tensor:
        # No KL term for MC-Dropout; returned for API symmetry.
        return torch.tensor(0.0)
