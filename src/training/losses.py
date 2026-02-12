"""Loss functions for ECG model training.

DenoiserLoss:  MSE + spectral L1 (preserves QRS frequency content)
HeatmapLoss:   weighted BCE with logits + Dice (handles class imbalance)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoiserLoss(nn.Module):
    """Combined time-domain MSE and frequency-domain spectral loss.

    Args:
        alpha: Weight for time-domain MSE loss (default 1.0).
        beta: Weight for spectral L1 loss (default 0.1).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predicted: Denoised output [B, 7, 2400].
            target: Clean ECG ground truth [B, 7, 2400].

        Returns:
            Scalar loss tensor.
        """
        mse = F.mse_loss(predicted, target)
        spectral = self._spectral_l1(predicted, target)
        return self.alpha * mse + self.beta * spectral

    @staticmethod
    def _spectral_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss on FFT magnitudes."""
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        return F.l1_loss(pred_fft.abs(), target_fft.abs())


class HeatmapLoss(nn.Module):
    """Combined BCE-with-logits and Dice loss for heatmap prediction.

    Args:
        alpha: Weight for BCE loss (default 1.0).
        beta: Weight for Dice loss (default 1.0).
        pos_weight: Positive class weight for BCE (default 50.0).
            Compensates for extreme class imbalance (~5% positive).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        pos_weight: float = 50.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined heatmap loss.

        Args:
            logits: Raw model output (before sigmoid) [B, 7, 9, 2400].
            targets: Ground truth heatmaps [B, 7, 9, 2400] in [0, 1].

        Returns:
            Scalar loss tensor.
        """
        pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw,
        )
        dice = self._dice_loss(logits, targets)
        return self.alpha * bce + self.beta * dice

    @staticmethod
    def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Soft Dice loss computed from logits."""
        probs = torch.sigmoid(logits)
        # Flatten spatial dims for Dice computation
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return 1.0 - dice
