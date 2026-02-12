"""U-Net heatmap decoder for fiducial point regression.

Takes encoded features from Phase 2 and produces per-lead, per-fiducial-type
heatmaps at the original signal resolution (2400 samples).

Input : [batch, 7, seq_len, d_model]  (fused features from Phase 2)
Output: [batch, 7, 9, 2400]           (9 fiducial heatmaps per lead, values in [0,1])

Fiducial types (channel order):
  0: P-onset   1: P-peak   2: P-offset
  3: QRS-onset 4: R-peak   5: QRS-offset
  6: T-onset   7: T-peak   8: T-offset
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


FIDUCIAL_NAMES = [
    "P_onset", "P_peak", "P_offset",
    "QRS_onset", "R_peak", "QRS_offset",
    "T_onset", "T_peak", "T_offset",
]
NUM_FIDUCIAL_TYPES = len(FIDUCIAL_NAMES)


class _DecoderBlock(nn.Module):
    """Upsample → Conv1d → BN → ReLU → Conv1d → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class HeatmapDecoder(nn.Module):
    """U-Net-style decoder producing fiducial point heatmaps.

    Args:
        d_model:           Input feature dimension from encoder (default 256).
        base_channels:     Base channel count for decoder (default 64).
        num_fiducial_types: Number of fiducial types (default 9).
        target_length:     Output temporal resolution (default 2400).
    """

    def __init__(
        self,
        d_model: int = 256,
        base_channels: int = 64,
        num_fiducial_types: int = NUM_FIDUCIAL_TYPES,
        target_length: int = 2400,
    ) -> None:
        super().__init__()
        self.target_length = target_length
        self.num_fiducial_types = num_fiducial_types

        # Project d_model → base_channels * 4 as bottleneck
        bc = base_channels
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, bc * 4),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks: progressively upsample
        self.dec1 = _DecoderBlock(bc * 4, bc * 2)
        self.dec2 = _DecoderBlock(bc * 2, bc)
        self.dec3 = _DecoderBlock(bc, bc)

        # Final 1x1 conv to fiducial channels
        self.head = nn.Conv1d(bc, num_fiducial_types, kernel_size=1)

    def forward(
        self, features: torch.Tensor, return_logits: bool = False,
    ) -> torch.Tensor:
        """Decode features into heatmaps.

        Args:
            features: ``[batch, 7, seq_len, d_model]`` from Phase 2 encoder.
            return_logits: If ``True``, return raw logits without sigmoid.
                Use this during training for numerically stable
                ``F.binary_cross_entropy_with_logits()``.

        Returns:
            Heatmaps of shape ``[batch, 7, 9, 2400]``.
            Values in [0, 1] when ``return_logits=False`` (default),
            unbounded logits when ``return_logits=True``.
        """
        B, num_leads, S, D = features.shape

        lead_heatmaps = []
        for lead_idx in range(num_leads):
            x = features[:, lead_idx]            # [B, S, D]
            x = self.input_proj(x)               # [B, S, bc*4]
            x = x.transpose(1, 2)                # [B, bc*4, S]

            # Upsample through decoder blocks
            x = self.dec1(x)                     # [B, bc*2, S*2]
            x = self.dec2(x)                     # [B, bc, S*4]
            x = self.dec3(x)                     # [B, bc, S*8]

            # Interpolate to exact target length
            x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)

            # Project to fiducial channels
            x = self.head(x)                     # [B, 9, 2400]
            if not return_logits:
                x = torch.sigmoid(x)

            lead_heatmaps.append(x)

        return torch.stack(lead_heatmaps, dim=1)  # [B, 7, 9, 2400]
