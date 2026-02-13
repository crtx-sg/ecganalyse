"""ECG-TransCovNet encoder adapter for the Phase 2 pipeline.

Wraps the ECG-TransCovNet architecture (CNN backbone with Selective Kernel
modules + Transformer encoder) as a per-lead feature encoder compatible with
the pipeline's ``[B, 7, seq_len, output_dim]`` contract.

The trained model checkpoint (``models/ecg_transcovnet/best_model.pt``) was
trained on all 7 leads simultaneously (in_channels=7).  For per-lead feature
extraction the encoder processes each lead independently through a shared
backbone (in_channels=1).  Weights from the trained checkpoint are transferred
where possible—all layers except the first Conv1d (whose input channels change
from 7 → 1) are loaded directly.  The first Conv1d is initialised by averaging
the trained weights across the input-channel axis.

Output contract (matches FoundationModelAdapter):
    Input:  [B, 7, 2400]
    Output: [B, 7, seq_len, output_dim]  where seq_len=150 for 2400-sample input
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default path to the trained checkpoint (relative to project root)
_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "ecg_transcovnet" / "best_model.pt"


# ─── Model components (same architecture as train_ecg_transcovnet.py) ─────────

class _SKConv(nn.Module):
    """Selective Kernel convolution block."""

    def __init__(self, in_ch: int, out_ch: int, M: int = 2, r: int = 16):
        super().__init__()
        d = max(in_ch // r, 32)
        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3 + i * 2, padding=1 + i),
                nn.BatchNorm1d(out_ch),
                nn.SiLU(inplace=True),
            )
            for i in range(M)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_ch, d)
        self.fcs = nn.ModuleList(nn.Linear(d, out_ch) for _ in range(M))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [conv(x) for conv in self.convs]
        feats_cat = torch.stack(feats, dim=1)
        s = self.gap(sum(feats)).squeeze(-1)
        z = self.fc(s)
        weights = torch.stack([fc(z) for fc in self.fcs], dim=1)
        attn = self.softmax(weights).unsqueeze(-1)
        return (feats_cat * attn).sum(dim=1)


class _CNNBackbone(nn.Module):
    """CNN feature extractor: 2400 → 150 tokens."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.sk_block = _SKConv(32, 64)
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.bottleneck = nn.Conv1d(128, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.sk_block(x)
        x = self.stage2(x)
        return self.bottleneck(x)


# ─── Phase 2 encoder ─────────────────────────────────────────────────────────

class ECGTransCovNetEncoder(nn.Module):
    """Per-lead feature encoder using ECG-TransCovNet architecture.

    Each of the 7 ECG leads is processed independently through a shared
    CNN backbone + Transformer encoder, producing per-lead temporal features
    at sequence length 150 (for 2400-sample / 12 s input).

    Args:
        embed_dim:           Internal embedding dimension of the TransCovNet backbone (default 128).
        output_dim:          Output feature dimension per time-step (default 256).
        nhead:               Number of attention heads in the Transformer encoder.
        num_encoder_layers:  Number of Transformer encoder layers.
        dim_feedforward:     Feed-forward dimension in Transformer encoder.
        dropout:             Dropout rate.
        signal_length:       Expected input signal length in samples (default 2400).
        weights_path:        Path to trained ECG-TransCovNet checkpoint.
                             If ``None``, uses default path if it exists.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        output_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        signal_length: int = 2400,
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Per-lead CNN backbone (in_channels=1)
        self.cnn_backbone = _CNNBackbone(in_channels=1, embed_dim=embed_dim)

        # Compute sequence length from backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 1, signal_length)
            seq_len = self.cnn_backbone(dummy).shape[2]
        self.seq_len = seq_len

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(seq_len, 1, embed_dim) * 0.02
        )

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        # Project to pipeline-expected output dimension
        if embed_dim != output_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(embed_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.output_proj = nn.Identity()

        # Load pretrained weights
        wp = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        if wp.exists():
            self._load_pretrained(wp)
        else:
            logger.warning(
                "TransCovNet weights not found at %s — using random initialisation", wp,
            )

    def _load_pretrained(self, weights_path: Path) -> None:
        """Load weights from trained 7-lead ECG-TransCovNet checkpoint.

        All layers except the first Conv1d are loaded directly.  The first
        Conv1d weights (shape ``[32, 7, 7]``) are averaged across the
        input-channel dimension to produce ``[32, 1, 7]`` for per-lead
        processing.
        """
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        src_sd = ckpt["model_state_dict"]

        # Map trained model keys → per-lead encoder keys
        KEY_MAP = {
            "cnn_backbone.": "cnn_backbone.",
            "positional_encoding": "positional_encoding",
            "encoder.": "encoder.",
        }

        loaded, skipped = 0, 0
        tgt_sd = self.state_dict()

        for src_key, src_val in src_sd.items():
            tgt_key = None
            for prefix in KEY_MAP:
                if src_key.startswith(prefix):
                    tgt_key = src_key
                    break
            if tgt_key is None or tgt_key not in tgt_sd:
                skipped += 1
                continue

            # Handle first conv layer: average 7 input channels → 1
            if tgt_key == "cnn_backbone.stage1.0.weight":
                # src: [32, 7, 7], tgt: [32, 1, 7]
                if src_val.shape[1] != tgt_sd[tgt_key].shape[1]:
                    src_val = src_val.mean(dim=1, keepdim=True)

            if src_val.shape == tgt_sd[tgt_key].shape:
                tgt_sd[tgt_key] = src_val
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(tgt_sd)
        logger.info(
            "Loaded %d parameters from TransCovNet checkpoint (%d skipped)", loaded, skipped,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 7-lead ECG into per-lead temporal features.

        Args:
            x: ``[B, 7, 2400]``.

        Returns:
            ``[B, 7, seq_len, output_dim]`` where ``seq_len=150``.
        """
        B, num_leads, T = x.shape

        # Reshape all leads into batch dimension for efficient processing
        x_flat = x.reshape(B * num_leads, 1, T)         # [B*7, 1, 2400]

        # CNN backbone
        features = self.cnn_backbone(x_flat)             # [B*7, embed_dim, seq_len]

        # Transformer encoder
        features = features.permute(2, 0, 1)             # [seq_len, B*7, embed_dim]
        features = features + self.positional_encoding
        memory = self.encoder(features)                  # [seq_len, B*7, embed_dim]

        # Reshape back to per-lead
        memory = memory.permute(1, 0, 2)                 # [B*7, seq_len, embed_dim]
        memory = self.output_proj(memory)                # [B*7, seq_len, output_dim]
        memory = memory.reshape(B, num_leads, self.seq_len, self.output_dim)

        return memory                                    # [B, 7, 150, output_dim]
