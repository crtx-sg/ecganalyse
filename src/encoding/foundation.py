"""Foundation model adapter for ECG feature encoding.

Optional wrapper that can substitute a pretrained ECG foundation model for
the Mamba + Swin dual-path encoder.  When no foundation model weights are
provided the adapter falls back to the dual-path encoding pipeline.

Supported ``model_type`` values:
    ``"dual_path"`` (default) — ECGMamba + Swin1DTransformer + DualPathFusion
    ``"transcovnet"`` — ECG-TransCovNet per-lead encoder (trained checkpoint)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from src.encoding.mamba import ECGMamba
from src.encoding.swin import Swin1DTransformer
from src.encoding.fusion import DualPathFusion

logger = logging.getLogger(__name__)


class FoundationModelAdapter(nn.Module):
    """Drop-in encoder that delegates to either a foundation model,
    an ECG-TransCovNet encoder, or the built-in Mamba + Swin + Fusion
    dual-path pipeline.

    Args:
        foundation_model: Optional pre-initialised encoder ``nn.Module``.
            Must accept ``[B, 7, 2400]`` and return ``[B, 7, S, D]``.
        output_dim: Expected output feature dimension.
        model_type: Encoder backend to use when *foundation_model* is ``None``.
            ``"dual_path"`` (default) or ``"transcovnet"``.
        transcovnet_kwargs: Keyword arguments forwarded to
            :class:`~src.encoding.transcovnet.ECGTransCovNetEncoder`.
        mamba_kwargs: Keyword arguments forwarded to :class:`ECGMamba`.
        swin_kwargs: Keyword arguments forwarded to :class:`Swin1DTransformer`.
        fusion_kwargs: Keyword arguments forwarded to :class:`DualPathFusion`.
    """

    def __init__(
        self,
        foundation_model: nn.Module | None = None,
        output_dim: int = 256,
        model_type: str = "dual_path",
        transcovnet_kwargs: dict | None = None,
        mamba_kwargs: dict | None = None,
        swin_kwargs: dict | None = None,
        fusion_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self._model_type = model_type

        if foundation_model is not None:
            self.foundation = foundation_model
            self.transcovnet = None
            self.mamba = None
            self.swin = None
            self.fusion = None
        elif model_type == "transcovnet":
            from src.encoding.transcovnet import ECGTransCovNetEncoder

            tk = transcovnet_kwargs or {}
            self.foundation = None
            self.transcovnet = ECGTransCovNetEncoder(
                output_dim=output_dim, **tk,
            )
            self.mamba = None
            self.swin = None
            self.fusion = None
            logger.info("Using ECG-TransCovNet encoder (output_dim=%d)", output_dim)
        else:
            self.foundation = None
            self.transcovnet = None
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

    @property
    def uses_transcovnet(self) -> bool:
        return self.transcovnet is not None

    @property
    def model_type(self) -> str:
        if self.foundation is not None:
            return "foundation"
        if self.transcovnet is not None:
            return "transcovnet"
        return "dual_path"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ECG input.

        Args:
            x: ``[batch, 7, 2400]``.

        Returns:
            ``[batch, 7, seq_len, output_dim]``.
        """
        if self.foundation is not None:
            return self.foundation(x)

        if self.transcovnet is not None:
            return self.transcovnet(x)

        mamba_out = self.mamba(x)
        swin_out = self.swin(x)
        return self.fusion(mamba_out, swin_out)
