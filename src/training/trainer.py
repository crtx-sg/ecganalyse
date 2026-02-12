"""ECG model trainer with 4-stage training pipeline.

Stage A: Train denoiser (self-supervised: noisy → clean)
Stage B: Train encoder + HeatmapDecoder (frozen denoiser)
Stage C: Fine-tune full pipeline with discriminative LR
Stage D: (optional) Train LeadGNN on multi-lead beat features
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.encoding.fusion import DualPathFusion
from src.encoding.mamba import ECGMamba
from src.encoding.swin import Swin1DTransformer
from src.prediction.heatmap import HeatmapDecoder
from src.preprocessing.denoiser import ECGDenoiser
from src.training.checkpointing import save_checkpoint, load_checkpoint, load_model_weights
from src.training.datasets import SyntheticECGTrainingDataset, create_validation_dataset
from src.training.losses import DenoiserLoss, HeatmapLoss
from src.training.metrics import DenoiserMetric, FiducialDetectionMetric, HeartRateMetric

logger = logging.getLogger(__name__)


def _custom_collate(batch: list[dict]) -> dict:
    """Custom collate that handles string fields."""
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


class ECGTrainer:
    """4-stage ECG model trainer.

    Args:
        output_dir: Directory for checkpoints and logs.
        device: Torch device (default auto-detect).
        use_amp: Enable automatic mixed precision (default True).
    """

    def __init__(
        self,
        output_dir: str | Path = "models/weights",
        device: Optional[str] = None,
        use_amp: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_amp = use_amp and self.device.type == "cuda"

        # Models
        self.denoiser = ECGDenoiser().to(self.device)
        self.mamba = ECGMamba().to(self.device)
        self.swin = Swin1DTransformer().to(self.device)
        self.fusion = DualPathFusion(
            mamba_dim=256,
            swin_dim=self.swin.output_dim,
        ).to(self.device)
        self.decoder = HeatmapDecoder().to(self.device)

    def train_stage_a(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epoch_size: int = 5000,
        val_size: int = 500,
        patience: int = 10,
        warmup_steps: int = 500,
        resume_path: Optional[str] = None,
    ) -> Path:
        """Stage A: Train denoiser (self-supervised).

        Returns:
            Path to best checkpoint.
        """
        logger.info("=== Stage A: Training Denoiser ===")

        train_ds = SyntheticECGTrainingDataset(
            epoch_size=epoch_size, base_seed=42,
            noise_levels=["low", "medium"],
        )
        val_ds = create_validation_dataset(
            size=val_size, noise_levels=["low", "medium"],
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=_custom_collate,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=_custom_collate,
        )

        criterion = DenoiserLoss()
        optimizer = AdamW(self.denoiser.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = epochs * len(train_loader)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps))

        start_epoch = 0
        best_val_loss = float("inf")

        if resume_path:
            meta = load_checkpoint(
                resume_path, model=self.denoiser,
                optimizer=optimizer, scheduler=scheduler,
            )
            start_epoch = meta["epoch"] + 1
            best_val_loss = meta["best_val_loss"]
            logger.info("Resuming Stage A from epoch %d", start_epoch)

        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        best_path = self.output_dir / "stage_a_best.pt"
        no_improve = 0
        step = 0

        for epoch in range(start_epoch, epochs):
            self.denoiser.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                ecg_noisy = batch["ecg_noisy"].to(self.device)
                ecg_clean = batch["ecg_clean"].to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    denoised = self.denoiser(ecg_noisy)
                    loss = criterion(denoised, ecg_clean)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                step += 1
                if step <= warmup_steps:
                    warmup_lr = lr * step / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr
                else:
                    scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validation
            val_loss = self._validate_denoiser(val_loader, criterion)
            logger.info(
                "Stage A Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                epoch + 1, epochs, avg_train_loss, val_loss,
            )

            # Checkpointing
            save_checkpoint(
                self.output_dir / "stage_a_latest.pt",
                model=self.denoiser, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_val_loss=best_val_loss, stage="A",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                save_checkpoint(
                    best_path,
                    model=self.denoiser, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, best_val_loss=best_val_loss, stage="A",
                )
                logger.info("New best val_loss=%.6f", best_val_loss)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        return best_path

    @torch.no_grad()
    def _validate_denoiser(
        self,
        val_loader: DataLoader,
        criterion: DenoiserLoss,
    ) -> float:
        self.denoiser.eval()
        total_loss = 0.0
        n = 0
        for batch in val_loader:
            ecg_noisy = batch["ecg_noisy"].to(self.device)
            ecg_clean = batch["ecg_clean"].to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                denoised = self.denoiser(ecg_noisy)
                loss = criterion(denoised, ecg_clean)
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def train_stage_b(
        self,
        epochs: int = 100,
        batch_size: int = 4,
        lr: float = 0.0005,
        weight_decay: float = 1e-4,
        epoch_size: int = 5000,
        val_size: int = 500,
        patience: int = 15,
        denoiser_checkpoint: Optional[str] = None,
        resume_path: Optional[str] = None,
    ) -> Path:
        """Stage B: Train encoder + decoder (frozen denoiser).

        Returns:
            Path to best checkpoint.
        """
        logger.info("=== Stage B: Training Encoder + Decoder ===")

        # Load pretrained denoiser
        if denoiser_checkpoint:
            load_model_weights(denoiser_checkpoint, self.denoiser)

        # Freeze denoiser
        for param in self.denoiser.parameters():
            param.requires_grad = False
        self.denoiser.eval()

        train_ds = SyntheticECGTrainingDataset(
            epoch_size=epoch_size, base_seed=1000,
            noise_levels=["clean", "low", "medium"],
        )
        val_ds = create_validation_dataset(
            size=val_size, noise_levels=["clean", "low", "medium"],
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=_custom_collate,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=_custom_collate,
        )

        criterion = HeatmapLoss()

        # Trainable parameters: encoder + decoder
        trainable_params = (
            list(self.mamba.parameters())
            + list(self.swin.parameters())
            + list(self.fusion.parameters())
            + list(self.decoder.parameters())
        )
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs))

        start_epoch = 0
        best_val_loss = float("inf")

        if resume_path:
            # Build a combined model for checkpoint loading
            combined = self._combined_pipeline_model()
            meta = load_checkpoint(
                resume_path, model=combined, optimizer=optimizer, scheduler=scheduler,
            )
            self._load_from_combined(combined)
            start_epoch = meta["epoch"] + 1
            best_val_loss = meta["best_val_loss"]

        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        best_path = self.output_dir / "stage_b_best.pt"
        no_improve = 0

        for epoch in range(start_epoch, epochs):
            self.mamba.train()
            self.swin.train()
            self.fusion.train()
            self.decoder.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                ecg_noisy = batch["ecg_noisy"].to(self.device)
                heatmaps_gt = batch["heatmaps"].to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    denoised = self.denoiser(ecg_noisy)
                    mamba_out = self.mamba(denoised)
                    swin_out = self.swin(denoised)
                    fused = self.fusion(mamba_out, swin_out)
                    logits = self.decoder(fused, return_logits=True)
                    loss = criterion(logits, heatmaps_gt)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            val_loss = self._validate_pipeline(val_loader, criterion)
            logger.info(
                "Stage B Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                epoch + 1, epochs, avg_train_loss, val_loss,
            )

            combined = self._combined_pipeline_model()
            save_checkpoint(
                self.output_dir / "stage_b_latest.pt",
                model=combined, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_val_loss=best_val_loss, stage="B",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                save_checkpoint(
                    best_path,
                    model=combined, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, best_val_loss=best_val_loss, stage="B",
                )
                logger.info("New best val_loss=%.6f", best_val_loss)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        return best_path

    @torch.no_grad()
    def _validate_pipeline(
        self,
        val_loader: DataLoader,
        criterion: HeatmapLoss,
    ) -> float:
        self.denoiser.eval()
        self.mamba.eval()
        self.swin.eval()
        self.fusion.eval()
        self.decoder.eval()
        total_loss = 0.0
        n = 0
        for batch in val_loader:
            ecg_noisy = batch["ecg_noisy"].to(self.device)
            heatmaps_gt = batch["heatmaps"].to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                denoised = self.denoiser(ecg_noisy)
                mamba_out = self.mamba(denoised)
                swin_out = self.swin(denoised)
                fused = self.fusion(mamba_out, swin_out)
                logits = self.decoder(fused, return_logits=True)
                loss = criterion(logits, heatmaps_gt)
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def train_stage_c(
        self,
        epochs: int = 50,
        batch_size: int = 4,
        lr: float = 0.0001,
        weight_decay: float = 1e-4,
        epoch_size: int = 5000,
        val_size: int = 500,
        patience: int = 15,
        checkpoint: Optional[str] = None,
        resume_path: Optional[str] = None,
    ) -> Path:
        """Stage C: Fine-tune full pipeline with discriminative LR.

        Returns:
            Path to best checkpoint.
        """
        logger.info("=== Stage C: Fine-tuning Full Pipeline ===")

        if checkpoint:
            combined = self._combined_pipeline_model()
            load_model_weights(checkpoint, combined)
            self._load_from_combined(combined)

        # Unfreeze denoiser
        for param in self.denoiser.parameters():
            param.requires_grad = True

        train_ds = SyntheticECGTrainingDataset(
            epoch_size=epoch_size, base_seed=2000,
            noise_levels=["clean", "low", "medium", "high"],
        )
        val_ds = create_validation_dataset(
            size=val_size, noise_levels=["clean", "low", "medium", "high"],
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=_custom_collate,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=_custom_collate,
        )

        criterion = HeatmapLoss()

        # Discriminative LR: denoiser 0.01x, encoder 0.1x, decoder 1x
        optimizer = AdamW([
            {"params": self.denoiser.parameters(), "lr": lr * 0.01},
            {"params": list(self.mamba.parameters()) + list(self.swin.parameters())
                       + list(self.fusion.parameters()), "lr": lr * 0.1},
            {"params": self.decoder.parameters(), "lr": lr},
        ], weight_decay=weight_decay)

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs))

        start_epoch = 0
        best_val_loss = float("inf")

        if resume_path:
            combined = self._combined_pipeline_model()
            meta = load_checkpoint(
                resume_path, model=combined, optimizer=optimizer, scheduler=scheduler,
            )
            self._load_from_combined(combined)
            start_epoch = meta["epoch"] + 1
            best_val_loss = meta["best_val_loss"]

        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        best_path = self.output_dir / "stage_c_best.pt"
        no_improve = 0

        for epoch in range(start_epoch, epochs):
            self.denoiser.train()
            self.mamba.train()
            self.swin.train()
            self.fusion.train()
            self.decoder.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                ecg_noisy = batch["ecg_noisy"].to(self.device)
                heatmaps_gt = batch["heatmaps"].to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    denoised = self.denoiser(ecg_noisy)
                    mamba_out = self.mamba(denoised)
                    swin_out = self.swin(denoised)
                    fused = self.fusion(mamba_out, swin_out)
                    logits = self.decoder(fused, return_logits=True)
                    loss = criterion(logits, heatmaps_gt)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            val_loss = self._validate_pipeline(val_loader, criterion)
            logger.info(
                "Stage C Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                epoch + 1, epochs, avg_train_loss, val_loss,
            )

            combined = self._combined_pipeline_model()
            save_checkpoint(
                self.output_dir / "stage_c_latest.pt",
                model=combined, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_val_loss=best_val_loss, stage="C",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                save_checkpoint(
                    best_path,
                    model=combined, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, best_val_loss=best_val_loss, stage="C",
                )
                logger.info("New best val_loss=%.6f", best_val_loss)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        return best_path

    # ------------------------------------------------------------------
    # Combined model helpers
    # ------------------------------------------------------------------

    def _combined_pipeline_model(self) -> nn.Module:
        """Create a nn.ModuleDict wrapping all pipeline models for checkpointing."""
        return nn.ModuleDict({
            "denoiser": self.denoiser,
            "mamba": self.mamba,
            "swin": self.swin,
            "fusion": self.fusion,
            "decoder": self.decoder,
        })

    def _load_from_combined(self, combined: nn.ModuleDict) -> None:
        """Copy weights from combined model dict back to individual models."""
        self.denoiser.load_state_dict(combined["denoiser"].state_dict())
        self.mamba.load_state_dict(combined["mamba"].state_dict())
        self.swin.load_state_dict(combined["swin"].state_dict())
        self.fusion.load_state_dict(combined["fusion"].state_dict())
        self.decoder.load_state_dict(combined["decoder"].state_dict())
