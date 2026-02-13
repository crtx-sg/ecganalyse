"""Neural condition classifier using the trained ECG-TransCovNet model.

Provides 16-class cardiac condition classification from the full 7-lead ECG
signal.  The classifier loads the trained checkpoint and runs inference to
produce condition predictions with per-class probabilities.

This enriches the Phase 3 / Phase 4 pipeline — the rule-based reasoning
engine can use the neural prediction to augment its rhythm classification,
especially for conditions that are hard to distinguish from measurements
alone (e.g. atrial fibrillation vs flutter, bundle-branch blocks).

Usage::

    classifier = ConditionClassifier()  # loads default weights
    result = classifier.classify(ecg_tensor)
    print(result.condition, result.confidence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoding.transcovnet import _CNNBackbone, _SKConv

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[2] / "models" / "ecg_transcovnet" / "best_model.pt"
)

# Condition names from the simulator (in enum order — matches training labels)
CONDITION_NAMES: list[str] = [
    "NORMAL_SINUS",
    "SINUS_BRADYCARDIA",
    "SINUS_TACHYCARDIA",
    "ATRIAL_FIBRILLATION",
    "ATRIAL_FLUTTER",
    "PAC",
    "SVT",
    "PVC",
    "VENTRICULAR_TACHYCARDIA",
    "VENTRICULAR_FIBRILLATION",
    "LBBB",
    "RBBB",
    "AV_BLOCK_1",
    "AV_BLOCK_2_TYPE1",
    "AV_BLOCK_2_TYPE2",
    "ST_ELEVATION",
]

# Map neural model condition names → pipeline rhythm classification names
CONDITION_TO_RHYTHM: dict[str, str] = {
    "NORMAL_SINUS": "normal_sinus_rhythm",
    "SINUS_BRADYCARDIA": "sinus_bradycardia",
    "SINUS_TACHYCARDIA": "sinus_tachycardia",
    "ATRIAL_FIBRILLATION": "atrial_fibrillation",
    "ATRIAL_FLUTTER": "atrial_flutter",
    "PAC": "premature_atrial_contractions",
    "SVT": "supraventricular_tachycardia",
    "PVC": "premature_ventricular_contractions",
    "VENTRICULAR_TACHYCARDIA": "ventricular_tachycardia",
    "VENTRICULAR_FIBRILLATION": "ventricular_fibrillation",
    "LBBB": "left_bundle_branch_block",
    "RBBB": "right_bundle_branch_block",
    "AV_BLOCK_1": "first_degree_av_block",
    "AV_BLOCK_2_TYPE1": "second_degree_av_block_type1",
    "AV_BLOCK_2_TYPE2": "second_degree_av_block_type2",
    "ST_ELEVATION": "st_elevation",
}


@dataclass
class ConditionPrediction:
    """Result from the neural condition classifier."""

    condition: str                          # e.g. "ATRIAL_FIBRILLATION"
    rhythm_label: str                       # e.g. "atrial_fibrillation"
    confidence: float                       # probability of the top prediction
    all_probabilities: dict[str, float]     # all 16 class probabilities
    top_k: list[tuple[str, float]] = field(default_factory=list)  # top 3 predictions


class _CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Decoder layer returning cross-attention weights."""

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, is_causal=tgt_is_causal,
            )
            out, attn = self.multihead_attn(
                self.norm2(x), memory, memory, need_weights=True,
            )
            x = x + self.dropout2(out)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, is_causal=tgt_is_causal)
            )
            out, attn = self.multihead_attn(x, memory, memory, need_weights=True)
            x = self.norm2(x + self.dropout2(out))
            x = self.norm3(x + self._ff_block(x))
        return x, attn


class _ECGTransCovNetFull(nn.Module):
    """Full ECG-TransCovNet model (classification head included).

    Architecture matches ``train_ecg_transcovnet.py`` exactly so that
    checkpoint weights can be loaded directly.
    """

    def __init__(
        self,
        num_classes: int = 16,
        in_channels: int = 7,
        signal_length: int = 2400,
        embed_dim: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.cnn_backbone = _CNNBackbone(in_channels, embed_dim)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, signal_length)
            seq_len = self.cnn_backbone(dummy).shape[2]
        self.seq_len = seq_len

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        self.decoder_layers = nn.ModuleList(
            _CustomTransformerDecoderLayer(
                d_model=embed_dim, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                batch_first=False,
            )
            for _ in range(num_decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.positional_encoding = nn.Parameter(
            torch.randn(seq_len, 1, embed_dim) * 0.02
        )
        self.object_queries = nn.Parameter(
            torch.randn(num_classes, 1, embed_dim) * 0.02
        )

        self.ffn_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn_backbone(x)
        features = features.permute(2, 0, 1)
        features = features + self.positional_encoding

        memory = self.encoder(features)

        B = x.shape[0]
        queries = self.object_queries.expand(-1, B, -1)
        dec = queries
        for layer in self.decoder_layers:
            dec, _ = layer(dec, memory)
        dec = self.decoder_norm(dec)

        dec = dec.permute(1, 0, 2)
        return self.ffn_head(dec).squeeze(-1)


class ConditionClassifier:
    """Neural cardiac condition classifier.

    Loads the trained ECG-TransCovNet checkpoint and classifies 7-lead
    ECG signals into one of 16 cardiac conditions.

    Args:
        weights_path: Path to the trained model checkpoint.
            Defaults to ``models/ecg_transcovnet/best_model.pt``.
        device: Torch device for inference (default: auto-detect).
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        wp = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Read checkpoint to extract model config
        if not wp.exists():
            raise FileNotFoundError(
                f"TransCovNet checkpoint not found at {wp}. "
                "Run train_ecg_transcovnet.py first."
            )

        ckpt = torch.load(wp, map_location=self.device, weights_only=False)
        args = ckpt.get("args", {})
        self.class_names = ckpt.get("class_names", CONDITION_NAMES)

        # Build model with matching config
        self.model = _ECGTransCovNetFull(
            num_classes=len(self.class_names),
            in_channels=len(ckpt.get("leads", ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"])),
            signal_length=2400,
            embed_dim=args.get("embed_dim", 128),
            nhead=args.get("nhead", 8),
            num_encoder_layers=args.get("num_encoder_layers", 3),
            num_decoder_layers=args.get("num_decoder_layers", 3),
            dim_feedforward=args.get("dim_feedforward", 512),
            dropout=args.get("dropout", 0.1),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Loaded ConditionClassifier (%d classes, device=%s)",
            len(self.class_names), self.device,
        )

    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> ConditionPrediction:
        """Classify a 7-lead ECG signal.

        Args:
            x: ``[B, 7, 2400]`` or ``[7, 2400]`` tensor.
                If unbatched, a batch dimension is added automatically.

        Returns:
            :class:`ConditionPrediction` for the first sample in the batch.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.to(self.device).float()

        # Per-lead z-score normalisation (must match training preprocessing)
        for b in range(x.shape[0]):
            for ch in range(x.shape[1]):
                mu = x[b, ch].mean()
                std = x[b, ch].std()
                if std > 1e-6:
                    x[b, ch] = (x[b, ch] - mu) / std

        logits = self.model(x)                           # [B, num_classes]
        probs = F.softmax(logits, dim=-1)

        # Take first sample in batch
        prob_vec = probs[0].cpu().numpy()
        pred_idx = int(prob_vec.argmax())
        condition = self.class_names[pred_idx]
        confidence = float(prob_vec[pred_idx])

        all_probs = {
            name: round(float(p), 4)
            for name, p in zip(self.class_names, prob_vec)
        }

        # Top-3 predictions
        sorted_indices = prob_vec.argsort()[::-1][:3]
        top_k = [
            (self.class_names[int(i)], round(float(prob_vec[i]), 4))
            for i in sorted_indices
        ]

        return ConditionPrediction(
            condition=condition,
            rhythm_label=CONDITION_TO_RHYTHM.get(condition, condition.lower()),
            confidence=round(confidence, 4),
            all_probabilities=all_probs,
            top_k=top_k,
        )
