"""Foundation model adapter for ECG feature encoding.

Optional wrapper that can substitute a pretrained ECG foundation model for
the Mamba + Swin dual-path encoder.  When no foundation model weights are
provided the adapter falls back to the dual-path encoding pipeline.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.encoding.mamba import ECGMamba
from src.encoding.swin import Swin1DTransformer
from src.encoding.fusion import DualPathFusion


class FoundationModelAdapter(nn.Module):
    """Drop-in encoder that delegates to either a foundation model or
    the built-in Mamba + Swin + Fusion dual-path pipeline.

    Args:
        foundation_model: Optional pre-initialised encoder ``nn.Module``.
            Must accept ``[B, 7, 2400]`` and return ``[B, 7, S, D]``.
        output_dim: Expected output feature dimension.
        mamba_kwargs: Keyword arguments forwarded to :class:`ECGMamba`.
        swin_kwargs: Keyword arguments forwarded to :class:`Swin1DTransformer`.
        fusion_kwargs: Keyword arguments forwarded to :class:`DualPathFusion`.
    """

    def __init__(
        self,
        foundation_model: nn.Module | None = None,
        output_dim: int = 256,
        mamba_kwargs: dict | None = None,
        swin_kwargs: dict | None = None,
        fusion_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        if foundation_model is not None:
            self.foundation = foundation_model
            self.mamba = None
            self.swin = None
            self.fusion = None
        else:
            self.foundation = None
            mk = mamba_kwargs or {}
            sk = swin_kwargs or {}
            fk = fusion_kwargs or {}

            self.mamba = ECGMamba(**mk)
            self.swin = Swin1DTransformer(**sk)

            # Resolve actual output dims for fusion
            mamba_dim = mk.get("d_model", 256)
            swin_dim = self.swin.output_dim
            self.fusion = DualPathFusion(
                mamba_dim=mamba_dim,
                swin_dim=swin_dim,
                output_dim=output_dim,
                **fk,
            )

    @property
    def uses_foundation(self) -> bool:
        return self.foundation is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ECG input.

        Args:
            x: ``[batch, 7, 2400]``.

        Returns:
            ``[batch, 7, seq_len, output_dim]``.
        """
        if self.foundation is not None:
            return self.foundation(x)

        mamba_out = self.mamba(x)
        swin_out = self.swin(x)
        return self.fusion(mamba_out, swin_out)
