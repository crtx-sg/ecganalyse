"""Evaluation metrics for ECG model training.

FiducialDetectionMetric: Per-channel detection rate, MAE, false positive rate
HeartRateMetric:         HR MAE/RMSE from R-peak detection
DenoiserMetric:          SNR improvement, RMSE, correlation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy.signal import find_peaks

from src.prediction.heatmap import FIDUCIAL_NAMES


@dataclass
class FiducialMetricResult:
    """Per-channel fiducial detection metrics."""

    detection_rate: dict[str, float] = field(default_factory=dict)
    mae_samples: dict[str, float] = field(default_factory=dict)
    mae_ms: dict[str, float] = field(default_factory=dict)
    false_positive_rate: dict[str, float] = field(default_factory=dict)
    overall_detection_rate: float = 0.0
    overall_mae_ms: float = 0.0


class FiducialDetectionMetric:
    """Evaluate fiducial point detection accuracy.

    Args:
        tolerance: Match tolerance in samples (default 8 = 40ms at 200Hz).
        peak_threshold: Minimum heatmap value to detect a peak (default 0.3).
        fs: Sampling frequency (default 200).
    """

    def __init__(
        self,
        tolerance: int = 8,
        peak_threshold: float = 0.3,
        fs: int = 200,
    ) -> None:
        self.tolerance = tolerance
        self.peak_threshold = peak_threshold
        self.fs = fs
        self._reset()

    def _reset(self) -> None:
        self._tp: dict[str, int] = {n: 0 for n in FIDUCIAL_NAMES}
        self._fn: dict[str, int] = {n: 0 for n in FIDUCIAL_NAMES}
        self._fp: dict[str, int] = {n: 0 for n in FIDUCIAL_NAMES}
        self._errors: dict[str, list[float]] = {n: [] for n in FIDUCIAL_NAMES}

    def reset(self) -> None:
        self._reset()

    def update(
        self,
        pred_heatmaps: np.ndarray,
        gt_heatmaps: np.ndarray,
    ) -> None:
        """Update metrics with a batch of predictions.

        Args:
            pred_heatmaps: Predicted heatmaps [9, 2400] (after sigmoid).
            gt_heatmaps: Ground truth heatmaps [9, 2400].
        """
        for ch_idx, name in enumerate(FIDUCIAL_NAMES):
            pred_peaks = self._find_peaks(pred_heatmaps[ch_idx])
            gt_peaks = self._find_peaks(gt_heatmaps[ch_idx])

            matched_gt = set()
            matched_pred = set()

            for pp in pred_peaks:
                best_dist = self.tolerance + 1
                best_gt = -1
                for gi, gp in enumerate(gt_peaks):
                    if gi in matched_gt:
                        continue
                    dist = abs(pp - gp)
                    if dist <= self.tolerance and dist < best_dist:
                        best_dist = dist
                        best_gt = gi
                if best_gt >= 0:
                    matched_gt.add(best_gt)
                    matched_pred.add(len(matched_pred))
                    self._tp[name] += 1
                    self._errors[name].append(float(best_dist))
                else:
                    self._fp[name] += 1

            self._fn[name] += len(gt_peaks) - len(matched_gt)

    def compute(self) -> FiducialMetricResult:
        """Compute aggregate metrics."""
        result = FiducialMetricResult()
        total_tp = 0
        total_gt = 0
        all_errors: list[float] = []

        for name in FIDUCIAL_NAMES:
            tp = self._tp[name]
            fn = self._fn[name]
            fp = self._fp[name]
            total = tp + fn

            result.detection_rate[name] = tp / total if total > 0 else 0.0
            result.false_positive_rate[name] = fp / (tp + fp) if (tp + fp) > 0 else 0.0

            errors = self._errors[name]
            result.mae_samples[name] = np.mean(errors).item() if errors else 0.0
            result.mae_ms[name] = result.mae_samples[name] / self.fs * 1000.0

            total_tp += tp
            total_gt += total
            all_errors.extend(errors)

        result.overall_detection_rate = total_tp / total_gt if total_gt > 0 else 0.0
        result.overall_mae_ms = (
            np.mean(all_errors).item() / self.fs * 1000.0 if all_errors else 0.0
        )
        return result

    def _find_peaks(self, heatmap: np.ndarray) -> list[int]:
        """Find peaks in a 1D heatmap."""
        peaks, _ = find_peaks(heatmap, height=self.peak_threshold, distance=10)
        return peaks.tolist()


@dataclass
class HeartRateMetricResult:
    """Heart rate detection metrics."""

    mae_bpm: float = 0.0
    rmse_bpm: float = 0.0
    n_samples: int = 0


class HeartRateMetric:
    """Evaluate heart rate estimation from R-peak detection.

    Args:
        fs: Sampling frequency (default 200).
        peak_threshold: Heatmap threshold (default 0.3).
        min_rr: Minimum R-R distance in samples (default 60).
    """

    def __init__(
        self,
        fs: int = 200,
        peak_threshold: float = 0.3,
        min_rr: int = 60,
    ) -> None:
        self.fs = fs
        self.peak_threshold = peak_threshold
        self.min_rr = min_rr
        self._errors: list[float] = []

    def reset(self) -> None:
        self._errors = []

    def update(
        self,
        pred_heatmaps: np.ndarray,
        gt_hr: float,
    ) -> None:
        """Update with a single prediction.

        Args:
            pred_heatmaps: Predicted heatmaps [9, 2400] (after sigmoid).
            gt_hr: Ground truth heart rate in BPM.
        """
        # R-peak is channel 4
        r_heatmap = pred_heatmaps[4]
        peaks, _ = find_peaks(
            r_heatmap, height=self.peak_threshold, distance=self.min_rr,
        )
        if len(peaks) < 2:
            return

        rr_intervals = np.diff(peaks) / self.fs  # seconds
        pred_hr = 60.0 / np.mean(rr_intervals)
        self._errors.append(abs(pred_hr - gt_hr))

    def compute(self) -> HeartRateMetricResult:
        """Compute aggregate HR metrics."""
        if not self._errors:
            return HeartRateMetricResult()
        errors = np.array(self._errors)
        return HeartRateMetricResult(
            mae_bpm=float(np.mean(errors)),
            rmse_bpm=float(np.sqrt(np.mean(errors ** 2))),
            n_samples=len(errors),
        )


@dataclass
class DenoiserMetricResult:
    """Denoiser evaluation metrics."""

    snr_improvement_db: float = 0.0
    rmse: float = 0.0
    correlation: float = 0.0
    n_samples: int = 0


class DenoiserMetric:
    """Evaluate denoiser performance.

    Computes SNR improvement (dB), RMSE, and correlation between
    denoised output and clean reference.
    """

    def __init__(self) -> None:
        self._snr_improvements: list[float] = []
        self._rmses: list[float] = []
        self._correlations: list[float] = []

    def reset(self) -> None:
        self._snr_improvements = []
        self._rmses = []
        self._correlations = []

    def update(
        self,
        denoised: np.ndarray,
        clean: np.ndarray,
        noisy: np.ndarray,
    ) -> None:
        """Update with a single sample.

        Args:
            denoised: Denoised signal [7, 2400].
            clean: Clean reference [7, 2400].
            noisy: Noisy input [7, 2400].
        """
        # Flatten across leads for aggregate metrics
        clean_flat = clean.ravel()
        denoised_flat = denoised.ravel()
        noisy_flat = noisy.ravel()

        signal_power = np.mean(clean_flat ** 2)
        if signal_power < 1e-10:
            return

        noise_before = np.mean((noisy_flat - clean_flat) ** 2)
        noise_after = np.mean((denoised_flat - clean_flat) ** 2)

        if noise_before > 1e-10 and noise_after > 1e-10:
            snr_before = 10 * np.log10(signal_power / noise_before)
            snr_after = 10 * np.log10(signal_power / noise_after)
            self._snr_improvements.append(snr_after - snr_before)

        self._rmses.append(float(np.sqrt(noise_after)))

        # Correlation
        if np.std(denoised_flat) > 1e-10 and np.std(clean_flat) > 1e-10:
            corr = np.corrcoef(denoised_flat, clean_flat)[0, 1]
            self._correlations.append(float(corr))

    def compute(self) -> DenoiserMetricResult:
        """Compute aggregate denoiser metrics."""
        if not self._rmses:
            return DenoiserMetricResult()
        return DenoiserMetricResult(
            snr_improvement_db=(
                float(np.mean(self._snr_improvements))
                if self._snr_improvements else 0.0
            ),
            rmse=float(np.mean(self._rmses)),
            correlation=(
                float(np.mean(self._correlations))
                if self._correlations else 0.0
            ),
            n_samples=len(self._rmses),
        )
