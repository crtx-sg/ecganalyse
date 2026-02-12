#!/usr/bin/env python
"""CLI for evaluating trained ECG models.

Usage:
    python scripts/evaluate.py --checkpoint models/weights/stage_c_best.pt
    python scripts/evaluate.py --checkpoint models/weights/stage_c_best.pt --num-samples 1000
    python scripts/evaluate.py --denoiser-only --checkpoint models/weights/stage_a_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import torch

from src.training.checkpointing import load_model_weights
from src.training.datasets import create_validation_dataset, _custom_collate
from src.training.metrics import (
    DenoiserMetric,
    FiducialDetectionMetric,
    HeartRateMetric,
)
from src.training.trainer import ECGTrainer

# Reuse the custom collate
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained ECG models")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument("--num-samples", type=int, default=500, help="Number of eval samples")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Torch device")
    parser.add_argument(
        "--denoiser-only", action="store_true",
        help="Evaluate denoiser only (Stage A checkpoint)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    trainer = ECGTrainer(device=args.device, use_amp=False)

    if args.denoiser_only:
        load_model_weights(args.checkpoint, trainer.denoiser)
        _evaluate_denoiser(trainer, args.num_samples, args.batch_size)
    else:
        combined = trainer._combined_pipeline_model()
        load_model_weights(args.checkpoint, combined)
        trainer._load_from_combined(combined)
        _evaluate_full(trainer, args.num_samples, args.batch_size)


@torch.no_grad()
def _evaluate_denoiser(
    trainer: ECGTrainer,
    num_samples: int,
    batch_size: int,
) -> None:
    val_ds = create_validation_dataset(
        size=num_samples,
        noise_levels=["low", "medium", "high"],
    )
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=_custom_collate,
    )

    metric = DenoiserMetric()
    trainer.denoiser.eval()

    for batch in loader:
        ecg_noisy = batch["ecg_noisy"].to(trainer.device)
        ecg_clean = batch["ecg_clean"].to(trainer.device)
        denoised = trainer.denoiser(ecg_noisy)

        # Per-sample metrics
        for i in range(ecg_noisy.shape[0]):
            metric.update(
                denoised[i].cpu().numpy(),
                ecg_clean[i].cpu().numpy(),
                ecg_noisy[i].cpu().numpy(),
            )

    result = metric.compute()
    print("\n=== Denoiser Evaluation ===")
    print(f"  SNR Improvement: {result.snr_improvement_db:.2f} dB")
    print(f"  RMSE:            {result.rmse:.6f}")
    print(f"  Correlation:     {result.correlation:.4f}")
    print(f"  Samples:         {result.n_samples}")


@torch.no_grad()
def _evaluate_full(
    trainer: ECGTrainer,
    num_samples: int,
    batch_size: int,
) -> None:
    val_ds = create_validation_dataset(
        size=num_samples,
        noise_levels=["clean", "low", "medium"],
    )
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=_custom_collate,
    )

    fid_metric = FiducialDetectionMetric()
    hr_metric = HeartRateMetric()
    denoiser_metric = DenoiserMetric()

    trainer.denoiser.eval()
    trainer.mamba.eval()
    trainer.swin.eval()
    trainer.fusion.eval()
    trainer.decoder.eval()

    for batch in loader:
        ecg_noisy = batch["ecg_noisy"].to(trainer.device)
        ecg_clean = batch["ecg_clean"].to(trainer.device)
        heatmaps_gt = batch["heatmaps"]
        hrs = batch["hr"]

        denoised = trainer.denoiser(ecg_noisy)
        mamba_out = trainer.mamba(denoised)
        swin_out = trainer.swin(denoised)
        fused = trainer.fusion(mamba_out, swin_out)
        heatmaps_pred = trainer.decoder(fused, return_logits=False)

        for i in range(ecg_noisy.shape[0]):
            # Denoiser metrics
            denoiser_metric.update(
                denoised[i].cpu().numpy(),
                ecg_clean[i].cpu().numpy(),
                ecg_noisy[i].cpu().numpy(),
            )

            # Use primary lead (ECG2, index 1) for fiducial and HR metrics
            pred_hm = heatmaps_pred[i, 1].cpu().numpy()  # [9, 2400]
            gt_hm = heatmaps_gt[i, 1].numpy()            # [9, 2400]

            fid_metric.update(pred_hm, gt_hm)
            hr_metric.update(pred_hm, float(hrs[i]))

    den_result = denoiser_metric.compute()
    fid_result = fid_metric.compute()
    hr_result = hr_metric.compute()

    print("\n=== Full Pipeline Evaluation ===")
    print("\n--- Denoiser ---")
    print(f"  SNR Improvement: {den_result.snr_improvement_db:.2f} dB")
    print(f"  Correlation:     {den_result.correlation:.4f}")

    print("\n--- Fiducial Detection ---")
    print(f"  Overall Det. Rate:  {fid_result.overall_detection_rate:.1%}")
    print(f"  Overall MAE:        {fid_result.overall_mae_ms:.1f} ms")
    for name in ["R_peak", "QRS_onset", "QRS_offset", "P_peak", "T_peak"]:
        rate = fid_result.detection_rate.get(name, 0)
        mae = fid_result.mae_ms.get(name, 0)
        print(f"  {name:15s}: det={rate:.1%}, MAE={mae:.1f} ms")

    print("\n--- Heart Rate ---")
    print(f"  MAE:  {hr_result.mae_bpm:.2f} bpm")
    print(f"  RMSE: {hr_result.rmse_bpm:.2f} bpm")
    print(f"  N:    {hr_result.n_samples}")


if __name__ == "__main__":
    main()
