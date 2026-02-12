#!/usr/bin/env python
"""CLI for training ECG analysis models.

Usage:
    python scripts/train.py --stage A --epochs 100
    python scripts/train.py --stage B --denoiser-checkpoint models/weights/stage_a_best.pt
    python scripts/train.py --stage C --checkpoint models/weights/stage_b_best.pt
    python scripts/train.py --stage all
    python scripts/train.py --stage B --resume models/weights/stage_b_latest.pt
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.training.trainer import ECGTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ECG analysis models")
    parser.add_argument(
        "--stage", required=True,
        choices=["A", "B", "C", "all"],
        help="Training stage: A (denoiser), B (encoder+decoder), C (fine-tune), all",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epoch-size", type=int, default=None, help="Virtual epoch size")
    parser.add_argument("--val-size", type=int, default=None, help="Validation set size")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--output-dir", type=str, default="models/weights", help="Output dir")
    parser.add_argument("--device", type=str, default=None, help="Torch device")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument(
        "--denoiser-checkpoint", type=str, default=None,
        help="Path to pretrained denoiser checkpoint (for Stage B)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to full pipeline checkpoint (for Stage C)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    trainer = ECGTrainer(
        output_dir=args.output_dir,
        device=args.device,
        use_amp=not args.no_amp,
    )

    def _kwargs(defaults: dict) -> dict:
        """Override defaults with CLI args where provided."""
        kw = dict(defaults)
        if args.epochs is not None:
            kw["epochs"] = args.epochs
        if args.batch_size is not None:
            kw["batch_size"] = args.batch_size
        if args.lr is not None:
            kw["lr"] = args.lr
        if args.epoch_size is not None:
            kw["epoch_size"] = args.epoch_size
        if args.val_size is not None:
            kw["val_size"] = args.val_size
        if args.patience is not None:
            kw["patience"] = args.patience
        return kw

    stage = args.stage.upper()

    if stage in ("A", "ALL"):
        kw = _kwargs({"epochs": 100, "batch_size": 16, "lr": 0.001})
        if args.resume and stage == "A":
            kw["resume_path"] = args.resume
        best_a = trainer.train_stage_a(**kw)
        logging.info("Stage A complete. Best checkpoint: %s", best_a)

    if stage in ("B", "ALL"):
        kw = _kwargs({"epochs": 100, "batch_size": 4, "lr": 0.0005})
        denoiser_ckpt = args.denoiser_checkpoint
        if stage == "ALL":
            denoiser_ckpt = str(trainer.output_dir / "stage_a_best.pt")
        if denoiser_ckpt:
            kw["denoiser_checkpoint"] = denoiser_ckpt
        if args.resume and stage == "B":
            kw["resume_path"] = args.resume
        best_b = trainer.train_stage_b(**kw)
        logging.info("Stage B complete. Best checkpoint: %s", best_b)

    if stage in ("C", "ALL"):
        kw = _kwargs({"epochs": 50, "batch_size": 4, "lr": 0.0001})
        ckpt = args.checkpoint
        if stage == "ALL":
            ckpt = str(trainer.output_dir / "stage_b_best.pt")
        if ckpt:
            kw["checkpoint"] = ckpt
        if args.resume and stage == "C":
            kw["resume_path"] = args.resume
        best_c = trainer.train_stage_c(**kw)
        logging.info("Stage C complete. Best checkpoint: %s", best_c)


if __name__ == "__main__":
    main()
