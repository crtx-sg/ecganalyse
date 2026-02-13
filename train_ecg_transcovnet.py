#!/usr/bin/env python3
"""Standalone ECG-TransCovNet training pipeline.

Generates diverse synthetic ECG data using the project's simulator (16 cardiac
conditions, 7 leads, 200 Hz, 12s duration) and trains the ECG-TransCovNet
hybrid CNN-Transformer model for arrhythmia classification.

Architecture (from the paper):
  - CNN backbone with Selective Kernel (SK) modules for local feature extraction
  - Transformer encoder-decoder with DETR-style object queries for global context
  - Focal Loss for class-imbalanced training

Usage:
    python train_ecg_transcovnet.py
    python train_ecg_transcovnet.py --epochs 150 --batch-size 64 --leads all
    python train_ecg_transcovnet.py --num-train 8000 --num-val 1600 --noise-level high
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project imports
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.simulator.conditions import Condition, CONDITION_REGISTRY
from src.simulator.ecg_simulator import ECGSimulator

# ─── Constants ────────────────────────────────────────────────────────────────

NUM_CLASSES = len(Condition)  # 16
CONDITION_LIST = list(Condition)
CONDITION_TO_IDX = {c: i for i, c in enumerate(CONDITION_LIST)}
CLASS_NAMES = [c.name for c in CONDITION_LIST]

SIGNAL_LENGTH = 2400  # 12 s * 200 Hz
ALL_LEADS = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]

# MIT-BIH-like proportions (from ecg_simulator_cli.py)
MIT_BIH_PROPORTIONS: dict[Condition, float] = {
    Condition.NORMAL_SINUS: 0.30,
    Condition.SINUS_BRADYCARDIA: 0.05,
    Condition.SINUS_TACHYCARDIA: 0.05,
    Condition.ATRIAL_FIBRILLATION: 0.10,
    Condition.ATRIAL_FLUTTER: 0.03,
    Condition.PAC: 0.05,
    Condition.SVT: 0.03,
    Condition.PVC: 0.10,
    Condition.VENTRICULAR_TACHYCARDIA: 0.05,
    Condition.VENTRICULAR_FIBRILLATION: 0.02,
    Condition.LBBB: 0.05,
    Condition.RBBB: 0.05,
    Condition.AV_BLOCK_1: 0.04,
    Condition.AV_BLOCK_2_TYPE1: 0.03,
    Condition.AV_BLOCK_2_TYPE2: 0.02,
    Condition.ST_ELEVATION: 0.03,
}


# ─── Data Generation ─────────────────────────────────────────────────────────

def _pick_condition(
    rng: np.random.Generator,
    proportions: dict[Condition, float] | None = None,
) -> Condition:
    """Sample a condition according to *proportions* (uniform if None)."""
    if proportions:
        conditions = list(proportions.keys())
        weights = np.array([proportions[c] for c in conditions])
    else:
        conditions = list(Condition)
        weights = np.ones(len(conditions))
    weights = weights / weights.sum()
    return conditions[rng.choice(len(conditions), p=weights)]


def generate_dataset(
    num_samples: int,
    leads: list[str],
    noise_level: str = "medium",
    proportions: dict[Condition, float] | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ECG signals using the simulator.

    Returns:
        signals: (N, num_leads, SIGNAL_LENGTH) float32
        labels:  (N,) int64
    """
    sim = ECGSimulator(seed=seed)
    signals = []
    labels = []

    for i in range(num_samples):
        condition = _pick_condition(sim._rng, proportions)
        ecg = sim.generate_ecg(condition, noise_level=noise_level)

        # Stack selected leads: (num_leads, 2400)
        signal = np.stack([ecg[lead] for lead in leads], axis=0)

        # Per-lead z-score normalisation
        for ch in range(signal.shape[0]):
            mu, std = signal[ch].mean(), signal[ch].std()
            if std > 1e-6:
                signal[ch] = (signal[ch] - mu) / std

        signals.append(signal.astype(np.float32))
        labels.append(CONDITION_TO_IDX[condition])

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    return np.array(signals), np.array(labels, dtype=np.int64)


def load_or_generate_data(
    cache_dir: str,
    num_train: int,
    num_val: int,
    leads: list[str],
    noise_level: str,
    seed: int,
    distribution: str = "balanced",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached data or generate new."""
    leads_tag = "-".join(leads)
    cache_path = (
        Path(cache_dir)
        / f"ecg_{leads_tag}_{noise_level}_{distribution}_t{num_train}_v{num_val}_s{seed}.npz"
    )

    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        d = np.load(cache_path)
        return d["train_X"], d["train_y"], d["val_X"], d["val_y"]

    balanced = {c: 1.0 / NUM_CLASSES for c in Condition}

    if distribution == "mit_bih":
        train_props = MIT_BIH_PROPORTIONS
        desc = "MIT-BIH proportions"
    else:
        train_props = balanced
        desc = "balanced"

    print(f"Generating {num_train} training samples ({desc})...")
    train_X, train_y = generate_dataset(
        num_train, leads, noise_level, train_props, seed,
    )

    print(f"Generating {num_val} validation samples (balanced)...")
    val_X, val_y = generate_dataset(
        num_val, leads, noise_level, balanced, seed + 10000,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y,
    )
    print(f"Cached data to {cache_path}")
    return train_X, train_y, val_X, val_y


# ─── Model Architecture ──────────────────────────────────────────────────────

class SKConv(nn.Module):
    """Selective Kernel convolution block (paper Figure 3a).

    Uses *M* parallel conv branches with different kernel sizes and an
    attention mechanism to dynamically weight them.
    """

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
        feats_cat = torch.stack(feats, dim=1)          # (B, M, C, L)
        s = self.gap(sum(feats)).squeeze(-1)            # (B, C)
        z = self.fc(s)                                  # (B, d)
        weights = torch.stack([fc(z) for fc in self.fcs], dim=1)  # (B, M, C)
        attn = self.softmax(weights).unsqueeze(-1)      # (B, M, C, 1)
        return (feats_cat * attn).sum(dim=1)            # (B, C, L)


class CNNBackbone(nn.Module):
    """CNN feature extractor adapted for 2400-sample signals.

    Down-sampling path:
        2400 -> [conv s2 + pool s2] -> 600
             -> [SK block]          -> 600
             -> [conv s2 + pool s2] -> 150
             -> [1x1 bottleneck]    -> 150  (seq_len for transformer)
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.sk_block = SKConv(32, 64)
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


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Decoder layer that also returns cross-attention weights."""

    def forward(
        self, tgt, memory,
        tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        tgt_is_causal=False, memory_is_causal=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, is_causal=tgt_is_causal,
            )
            cross_out, cross_attn = self._mha_custom(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask,
            )
            x = x + cross_out
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, is_causal=tgt_is_causal)
            )
            cross_out, cross_attn = self._mha_custom(
                x, memory, memory_mask, memory_key_padding_mask,
            )
            x = self.norm2(x + cross_out)
            x = self.norm3(x + self._ff_block(x))
        return x, cross_attn

    def _mha_custom(self, x, mem, attn_mask, key_padding_mask):
        out, attn = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask, need_weights=True,
        )
        return self.dropout2(out), attn


class ECGTransCovNet(nn.Module):
    """Hybrid CNN-Transformer for ECG arrhythmia classification.

    Pipeline:
        raw signal  ->  CNN backbone (local features)
                    ->  Transformer encoder (global context)
                    ->  Transformer decoder with DETR object queries
                    ->  per-query FFN  ->  class logits
    """

    def __init__(
        self,
        num_classes: int = 16,
        in_channels: int = 1,
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

        # CNN backbone
        self.cnn_backbone = CNNBackbone(in_channels, embed_dim)

        # Determine the sequence length after CNN (data-driven)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, signal_length)
            seq_len = self.cnn_backbone(dummy).shape[2]
        self.seq_len = seq_len

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        # Transformer decoder (custom layers for attention extraction)
        self.decoder_layers = nn.ModuleList(
            CustomTransformerDecoderLayer(
                d_model=embed_dim, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                batch_first=False,
            )
            for _ in range(num_decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Learnable positional encoding & object queries
        self.positional_encoding = nn.Parameter(
            torch.randn(seq_len, 1, embed_dim) * 0.02
        )
        self.object_queries = nn.Parameter(
            torch.randn(num_classes, 1, embed_dim) * 0.02
        )

        # Classification head (per-query)
        self.ffn_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn_backbone(x)                        # (B, D, S)
        features = features.permute(2, 0, 1)                  # (S, B, D)
        features = features + self.positional_encoding

        memory = self.encoder(features)                        # (S, B, D)

        B = x.shape[0]
        queries = self.object_queries.expand(-1, B, -1)        # (C, B, D)
        dec = queries
        for layer in self.decoder_layers:
            dec, _ = layer(dec, memory)
        dec = self.decoder_norm(dec)                           # (C, B, D)

        dec = dec.permute(1, 0, 2)                             # (B, C, D)
        return self.ffn_head(dec).squeeze(-1)                  # (B, C)

    def forward_with_attention(
        self, x: torch.Tensor, layer_idx: int = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass that also returns cross-attention weights."""
        features = self.cnn_backbone(x).permute(2, 0, 1)
        features = features + self.positional_encoding
        memory = self.encoder(features)

        B = x.shape[0]
        dec = self.object_queries.expand(-1, B, -1)
        captured = None
        for i, layer in enumerate(self.decoder_layers):
            dec, attn = layer(dec, memory)
            target = layer_idx if layer_idx >= 0 else len(self.decoder_layers) - 1
            if i == target:
                captured = attn

        dec = self.decoder_norm(dec).permute(1, 0, 2)
        logits = self.ffn_head(dec).squeeze(-1)
        return logits, captured


# ─── Loss Function ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification (paper Section 3.5)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


# ─── Training Utilities ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(X)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_detailed(model, loader, device):
    """Per-class precision / recall / specificity / F1 + confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        preds = model(X).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro = defaultdict(float)
    per_class = {}
    for i in range(NUM_CLASSES):
        tp = int(((all_preds == i) & (all_labels == i)).sum())
        fp = int(((all_preds == i) & (all_labels != i)).sum())
        fn = int(((all_preds != i) & (all_labels == i)).sum())
        tn = int(((all_preds != i) & (all_labels != i)).sum())

        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

        per_class[CLASS_NAMES[i]] = dict(
            precision=prec, recall=rec, specificity=spec, f1=f1, support=tp + fn,
        )
        for k, v in [("precision", prec), ("recall", rec), ("specificity", spec), ("f1", f1)]:
            macro[k] += v

    macro = {k: v / NUM_CLASSES for k, v in macro.items()}
    macro["accuracy"] = float((all_preds == all_labels).mean())

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    return dict(macro), per_class, cm


# ─── Plotting ─────────────────────────────────────────────────────────────────

def load_hdf5_test_samples(
    test_dir: str,
    leads: list[str],
) -> list[Tuple[str, np.ndarray, int]]:
    """Load test samples from HDF5 files generated by generate_conditions_test_set.py.

    Returns:
        List of (label, signal, class_index) tuples.
    """
    import h5py

    # Map file label prefixes to Condition enum
    LABEL_TO_CONDITION = {
        "sinus_brady": Condition.SINUS_BRADYCARDIA,
        "normal_sinus": Condition.NORMAL_SINUS,
        "sinus_tachy": Condition.SINUS_TACHYCARDIA,
        "afib": Condition.ATRIAL_FIBRILLATION,
        "aflutter": Condition.ATRIAL_FLUTTER,
        "svt": Condition.SVT,
        "pvc": Condition.PVC,
        "vtach": Condition.VENTRICULAR_TACHYCARDIA,
        "lbbb": Condition.LBBB,
        "rbbb": Condition.RBBB,
        "avblock1": Condition.AV_BLOCK_1,
        "avblock2t1": Condition.AV_BLOCK_2_TYPE1,
        "ste": Condition.ST_ELEVATION,
    }

    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Warning: test directory {test_dir} does not exist")
        return []

    lead_order = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]
    samples = []

    for h5_file in sorted(test_path.glob("*.h5")):
        fname = h5_file.stem  # e.g. "afib_120_clean"

        # Find matching condition from prefix
        condition = None
        for prefix, cond in LABEL_TO_CONDITION.items():
            if fname.startswith(prefix):
                condition = cond
                break
        if condition is None:
            print(f"  Skipping {h5_file.name} (unknown condition)")
            continue

        class_idx = CONDITION_TO_IDX[condition]

        with h5py.File(h5_file, "r") as f:
            # Each file has event_1001
            for event_key in sorted(f.keys()):
                if not event_key.startswith("event_"):
                    continue
                event = f[event_key]
                if "ecg" not in event:
                    continue

                ecg_group = event["ecg"]
                # Stack leads
                lead_arrays = []
                for lead in leads:
                    if lead in ecg_group:
                        lead_arrays.append(ecg_group[lead][:])
                    else:
                        # Try matching from lead_order
                        idx = lead_order.index(lead) if lead in lead_order else -1
                        if idx >= 0 and str(idx) in ecg_group:
                            lead_arrays.append(ecg_group[str(idx)][:])
                        else:
                            lead_arrays.append(np.zeros(SIGNAL_LENGTH, dtype=np.float32))

                signal = np.stack(lead_arrays, axis=0)

                # Per-lead z-score (same as training)
                for ch in range(signal.shape[0]):
                    mu, std = signal[ch].mean(), signal[ch].std()
                    if std > 1e-6:
                        signal[ch] = (signal[ch] - mu) / std

                samples.append((fname, signal.astype(np.float32), class_idx))

    return samples


@torch.no_grad()
def evaluate_hdf5_test(
    model: nn.Module,
    test_dir: str,
    leads: list[str],
    device: torch.device,
) -> None:
    """Evaluate trained model on HDF5 test files and print per-sample results."""
    model.eval()
    samples = load_hdf5_test_samples(test_dir, leads)
    if not samples:
        print(f"No test samples found in {test_dir}")
        return

    print(f"\n=== HDF5 Test Evaluation ({len(samples)} samples from {test_dir}) ===")
    print(f"  {'File':<30s} {'True Label':<28s} {'Predicted':<28s} {'Conf':>6s} {'Match':>5s}")
    print("  " + "-" * 99)

    correct = 0
    for fname, signal, true_idx in samples:
        x = torch.from_numpy(signal).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        conf = probs[pred_idx].item()
        match = pred_idx == true_idx
        if match:
            correct += 1

        true_name = CLASS_NAMES[true_idx]
        pred_name = CLASS_NAMES[pred_idx]
        mark = "OK" if match else "MISS"
        print(f"  {fname:<30s} {true_name:<28s} {pred_name:<28s} {conf:6.3f} {mark:>5s}")

    acc = correct / len(samples) if samples else 0
    print(f"\n  Test accuracy: {correct}/{len(samples)} = {acc:.1%}")


def save_training_curves(history: dict, path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss Curves")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy Curves")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, names: list[str], path: str):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(cm.shape[1]), yticks=range(cm.shape[0]),
        xticklabels=names, yticklabels=names,
        title="Confusion Matrix", ylabel="True", xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=6,
            )
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train ECG-TransCovNet on synthetic ECG data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    g = p.add_argument_group("data")
    g.add_argument("--num-train", type=int, default=16000)
    g.add_argument("--num-val", type=int, default=3200)
    g.add_argument(
        "--leads", type=str, default="all",
        help="Comma-separated lead names, or 'all' for all 7 leads",
    )
    g.add_argument("--noise-level", type=str, default="clean",
                   choices=["clean", "low", "medium", "high"])
    g.add_argument("--distribution", type=str, default="balanced",
                   choices=["balanced", "mit_bih"],
                   help="Training data distribution (balanced or mit_bih)")
    g.add_argument("--cache-dir", type=str, default="data/training_cache")
    g.add_argument("--test-dir", type=str, default=None,
                   help="Directory with HDF5 test files for post-training evaluation")

    # Model
    g = p.add_argument_group("model")
    g.add_argument("--embed-dim", type=int, default=128)
    g.add_argument("--nhead", type=int, default=8)
    g.add_argument("--num-encoder-layers", type=int, default=3)
    g.add_argument("--num-decoder-layers", type=int, default=3)
    g.add_argument("--dim-feedforward", type=int, default=512)
    g.add_argument("--dropout", type=float, default=0.1)

    # Training
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--warmup-epochs", type=int, default=5)
    g.add_argument("--patience", type=int, default=20)
    g.add_argument("--seed", type=int, default=42)

    # Output
    g = p.add_argument_group("output")
    g.add_argument("--output-dir", type=str, default="models/ecg_transcovnet")
    return p


def main():
    args = build_parser().parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve leads
    if args.leads.lower() == "all":
        leads = ALL_LEADS
    else:
        leads = [l.strip() for l in args.leads.split(",")]
    in_channels = len(leads)
    print(f"Using {in_channels} lead(s): {leads}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n=== Data Generation ===")
    t0 = time.time()
    train_X, train_y, val_X, val_y = load_or_generate_data(
        args.cache_dir, args.num_train, args.num_val, leads, args.noise_level, args.seed,
        distribution=args.distribution,
    )
    print(f"Data ready in {time.time() - t0:.1f}s")
    print(f"Train: {train_X.shape}  Val: {val_X.shape}")
    print(f"Train dist: {dict(sorted(Counter(train_y.tolist()).items()))}")
    print(f"Val   dist: {dict(sorted(Counter(val_y.tolist()).items()))}")

    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n=== Model ===")
    model = ECGTransCovNet(
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        signal_length=SIGNAL_LENGTH,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"CNN output sequence length: {model.seq_len}")

    # Quick sanity check
    with torch.no_grad():
        dummy = torch.randn(2, in_channels, SIGNAL_LENGTH, device=device)
        out = model(dummy)
        assert out.shape == (2, NUM_CLASSES), f"Unexpected shape {out.shape}"
    print("Forward-pass sanity check passed.")

    # ── Training setup ────────────────────────────────────────────────────
    loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup = args.warmup_epochs
    total = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n=== Training ({args.epochs} epochs, patience={args.patience}) ===")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_ctr = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        vl_loss, vl_acc = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": vl_acc,
                    "val_loss": vl_loss,
                    "args": vars(args),
                    "leads": leads,
                    "class_names": CLASS_NAMES,
                },
                output_dir / "best_model.pt",
            )
            marker = "  *best*"
        else:
            patience_ctr += 1

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train {tr_loss:.4f} / {tr_acc:.4f} | "
            f"Val {vl_loss:.4f} / {vl_acc:.4f} | "
            f"LR {lr_now:.2e} | {dt:.1f}s{marker}"
        )

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping (no improvement for {args.patience} epochs)")
            break

    # ── Final evaluation ──────────────────────────────────────────────────
    print(f"\n=== Final Evaluation (best val acc: {best_val_acc:.4f}) ===")
    ckpt = torch.load(output_dir / "best_model.pt", weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    macro, per_class, cm = evaluate_detailed(model, val_loader, device)

    print(f"\nMacro-averaged Metrics:")
    for k in ("accuracy", "precision", "recall", "specificity", "f1"):
        print(f"  {k:<14s}: {macro[k]:.4f}")

    print(f"\nPer-class Metrics:")
    print(f"  {'Condition':<28s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} {'F1':>6s} {'N':>5s}")
    print("  " + "-" * 57)
    for name in CLASS_NAMES:
        m = per_class[name]
        print(
            f"  {name:<28s} {m['precision']:6.3f} {m['recall']:6.3f} "
            f"{m['specificity']:6.3f} {m['f1']:6.3f} {m['support']:5d}"
        )

    # ── Save outputs ──────────────────────────────────────────────────────
    save_training_curves(history, str(output_dir / "training_curves.png"))
    save_confusion_matrix(cm, CLASS_NAMES, str(output_dir / "confusion_matrix.png"))

    # Final model with all metadata
    torch.save(
        {
            "epoch": ckpt["epoch"],
            "model_state_dict": model.state_dict(),
            "val_acc": best_val_acc,
            "metrics": macro,
            "per_class_metrics": per_class,
            "confusion_matrix": cm,
            "class_names": CLASS_NAMES,
            "leads": leads,
            "args": vars(args),
        },
        output_dir / "final_model.pt",
    )

    # ── HDF5 test evaluation ─────────────────────────────────────────────
    if args.test_dir:
        evaluate_hdf5_test(model, args.test_dir, leads, device)

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  best_model.pt       - checkpoint with best val accuracy")
    print(f"  final_model.pt      - checkpoint with full evaluation metadata")
    print(f"  training_curves.png - loss and accuracy plots")
    print(f"  confusion_matrix.png - per-class confusion matrix")
    print("Done!")


if __name__ == "__main__":
    main()
