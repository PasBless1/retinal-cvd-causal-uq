"""
Bayesian U-Net for retinal vessel segmentation with predictive uncertainty.

Weights in the convolutional layers are treated as Gaussian random variables
with learned mean and log-variance (mean-field variational inference). At
inference we draw `mc_samples` weight samples; the mean of the sigmoid outputs
is the vessel probability map and the standard deviation is the epistemic
uncertainty map.

The KL term of the ELBO is added to the loss so training maximises a proper
variational objective rather than just minimising reconstruction error.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianConv2d(nn.Module):
    """Convolution with a Gaussian variational posterior over weights."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 prior_std: float = 1.0):
        super().__init__()
        self.in_ch, self.out_ch, self.kernel = in_ch, out_ch, kernel
        self.prior_std = prior_std

        # Variational parameters: weight ~ N(mu, softplus(rho)^2)
        self.weight_mu = nn.Parameter(
            torch.empty(out_ch, in_ch, kernel, kernel).normal_(0, 0.05)
        )
        self.weight_rho = nn.Parameter(
            torch.full((out_ch, in_ch, kernel, kernel), -5.0)
        )
        self.bias_mu = nn.Parameter(torch.zeros(out_ch))
        self.bias_rho = nn.Parameter(torch.full((out_ch,), -5.0))

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-6

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            w_std = self._softplus(self.weight_rho)
            b_std = self._softplus(self.bias_rho)
            w = self.weight_mu + w_std * torch.randn_like(w_std)
            b = self.bias_mu + b_std * torch.randn_like(b_std)
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.conv2d(x, w, b, padding=self.kernel // 2)

    def kl_divergence(self) -> torch.Tensor:
        """KL( q(w) || N(0, prior_std^2) ) summed over all parameters."""
        kl = 0.0
        for mu, rho in ((self.weight_mu, self.weight_rho),
                        (self.bias_mu, self.bias_rho)):
            std = self._softplus(rho)
            var = std ** 2
            prior_var = self.prior_std ** 2
            kl = kl + 0.5 * torch.sum(
                (var + mu ** 2) / prior_var
                - 1.0
                - torch.log(var)
                + math.log(prior_var)
            )
        return kl


class _Block(nn.Module):
    """Two Bayesian convs + ReLU, the standard U-Net double-conv block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1 = BayesianConv2d(in_ch, out_ch)
        self.c2 = BayesianConv2d(out_ch, out_ch)

    def forward(self, x, sample=True):
        x = F.relu(self.c1(x, sample))
        x = F.relu(self.c2(x, sample))
        return x

    def kl(self):
        return self.c1.kl_divergence() + self.c2.kl_divergence()


class BayesianUNet(nn.Module):
    """
    Compact 3-level Bayesian U-Net.

    Returns a dict with keys: mask (mean prob), uncertainty (std), samples.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = _Block(in_ch, base)
        self.enc2 = _Block(base, base * 2)
        self.enc3 = _Block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _Block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _Block(base * 2, base)
        self.head = BayesianConv2d(base, out_ch, kernel=1)

    def _forward_once(self, x: torch.Tensor, sample: bool) -> torch.Tensor:
        e1 = self.enc1(x, sample)
        e2 = self.enc2(self.pool(e1), sample)
        e3 = self.enc3(self.pool(e2), sample)

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), sample)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), sample)
        return self.head(d1, sample)

    def forward(self, x: torch.Tensor,
                mc_samples: int = 1) -> Dict[str, torch.Tensor]:
        if self.training:
            # Single sampled pass during training (stochastic ELBO).
            logits = self._forward_once(x, sample=True)
            return {"logits": logits, "mask": torch.sigmoid(logits)}

        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                preds.append(torch.sigmoid(self._forward_once(x, sample=True)))
        stacked = torch.stack(preds, dim=0)          # [S, B, 1, H, W]
        return {
            "mask": stacked.mean(0),
            "uncertainty": stacked.std(0),
            "samples": stacked,
        }

    def kl_divergence(self) -> torch.Tensor:
        return (
            self.enc1.kl() + self.enc2.kl() + self.enc3.kl()
            + self.dec2.kl() + self.dec1.kl() + self.head.kl_divergence()
        )


def elbo_loss(logits: torch.Tensor, target: torch.Tensor,
              kl: torch.Tensor, dataset_size: int) -> torch.Tensor:
    """
    Negative ELBO: BCE reconstruction term + KL scaled by 1/N.

    Scaling the KL by the dataset size keeps the variational objective
    correctly weighted when optimising over mini-batches.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + kl / max(dataset_size, 1)
