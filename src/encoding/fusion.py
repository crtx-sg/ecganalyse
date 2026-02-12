"""Dual-path feature fusion: combine Mamba and Swin encoder outputs.

Accepts two feature tensors with potentially different sequence lengths and
feature dimensions, interpolates to a common sequence length, concatenates
along the feature axis, and applies a learned projection.

Output: [batch, 7, seq_fused, output_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathFusion(nn.Module):
    """Fuse Mamba (global context) and Swin (local morphology) features.

    Args:
        mamba_dim:  Feature dimension of Mamba output.
        swin_dim:   Feature dimension of Swin output.
        output_dim: Fused feature dimension (default 256).
        strategy:   ``"concat_project"`` (default) or ``"cross_attention"``.
    """

    def __init__(
        self,
        mamba_dim: int = 256,
        swin_dim: int = 1024,
        output_dim: int = 256,
        strategy: str = "concat_project",
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.strategy = strategy

        if strategy == "concat_project":
            self.proj = nn.Sequential(
                nn.Linear(mamba_dim + swin_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
        elif strategy == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=output_dim, num_heads=8, batch_first=True,
            )
            self.proj_mamba = nn.Linear(mamba_dim, output_dim)
            self.proj_swin = nn.Linear(swin_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def forward(
        self,
        mamba_out: torch.Tensor,
        swin_out: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse encoder outputs.

        Args:
            mamba_out: [batch, 7, S1, D1] — Mamba features.
            swin_out:  [batch, 7, S2, D2] — Swin features.

        Returns:
            Fused tensor of shape [batch, 7, S_fused, output_dim].
        """
        B, num_leads, S1, D1 = mamba_out.shape
        _, _, S2, D2 = swin_out.shape

        # Use the longer sequence length as the target
        S_fused = max(S1, S2)

        lead_outputs = []
        for lead_idx in range(num_leads):
            m = mamba_out[:, lead_idx]   # [B, S1, D1]
            s = swin_out[:, lead_idx]    # [B, S2, D2]

            # Interpolate to common length
            if S1 != S_fused:
                m = self._interpolate(m, S_fused)
            if S2 != S_fused:
                s = self._interpolate(s, S_fused)

            if self.strategy == "concat_project":
                fused = torch.cat([m, s], dim=-1)    # [B, S, D1+D2]
                fused = self.proj(fused)              # [B, S, output_dim]
            else:  # cross_attention
                m_proj = self.proj_mamba(m)           # [B, S, output_dim]
                s_proj = self.proj_swin(s)            # [B, S, output_dim]
                fused, _ = self.cross_attn(m_proj, s_proj, s_proj)
                fused = self.norm(fused + m_proj)     # residual

            lead_outputs.append(fused)

        return torch.stack(lead_outputs, dim=1)  # [B, 7, S_fused, output_dim]

    @staticmethod
    def _interpolate(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Interpolate sequence dimension: [B, S, D] → [B, target_len, D]."""
        # F.interpolate works on [B, C, L] — treat D as channels
        x_t = x.transpose(1, 2)  # [B, D, S]
        x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        return x_t.transpose(1, 2)  # [B, target_len, D]
