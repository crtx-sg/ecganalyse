"""Save, load, and resume training state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    epoch: int,
    best_val_loss: float,
    stage: str,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path: File path for the checkpoint.
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: Optional LR scheduler state.
        epoch: Current epoch number.
        best_val_loss: Best validation loss so far.
        stage: Training stage identifier (e.g. "A", "B", "C").
        extra: Optional extra metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "stage": stage,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        state["extra"] = extra

    torch.save(state, path)
    logger.info("Saved checkpoint to %s (epoch %d, val_loss=%.6f)", path, epoch, best_val_loss)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        map_location: Device mapping for torch.load.

    Returns:
        Dictionary with checkpoint metadata (epoch, best_val_loss, stage, extra).
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded model weights from %s", path)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "stage": checkpoint.get("stage", ""),
        "extra": checkpoint.get("extra", {}),
    }


def load_model_weights(
    path: str | Path,
    model: nn.Module,
    strict: bool = True,
    map_location: str = "cpu",
) -> None:
    """Load only model weights from a checkpoint (no optimizer/scheduler).

    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        strict: Whether to require exact key matching.
        map_location: Device mapping.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    logger.info("Loaded model weights from %s (strict=%s)", path, strict)
