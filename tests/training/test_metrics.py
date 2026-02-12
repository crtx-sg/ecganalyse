"""Tests for training evaluation metrics."""

import numpy as np
import pytest

from src.training.metrics import (
    DenoiserMetric,
    FiducialDetectionMetric,
    HeartRateMetric,
)


class TestFiducialDetectionMetric:
    def test_perfect_prediction(self):
        """When pred == gt, detection rate should be 1.0."""
        metric = FiducialDetectionMetric(tolerance=8)

        # Create a heatmap with known peaks
        hm = np.zeros((9, 2400), dtype=np.float32)
        # R-peak at sample 200, 600, 1000, 1400
        for pos in [200, 600, 1000, 1400]:
            x = np.arange(2400, dtype=np.float32)
            hm[4] += np.exp(-0.5 * ((x - pos) / 4.0) ** 2)

        metric.update(hm, hm)
        result = metric.compute()

        assert result.detection_rate["R_peak"] == pytest.approx(1.0)
        assert result.mae_samples["R_peak"] == pytest.approx(0.0)

    def test_no_predictions(self):
        """When pred is all zeros, detection rate should be 0."""
        metric = FiducialDetectionMetric()

        gt = np.zeros((9, 2400), dtype=np.float32)
        gt[4][200] = 1.0
        gt[4][600] = 1.0

        pred = np.zeros((9, 2400), dtype=np.float32)

        metric.update(pred, gt)
        result = metric.compute()

        assert result.detection_rate["R_peak"] == 0.0

    def test_reset(self):
        metric = FiducialDetectionMetric()
        hm = np.zeros((9, 2400), dtype=np.float32)
        hm[4][200] = 1.0
        metric.update(hm, hm)
        metric.reset()
        result = metric.compute()
        assert result.overall_detection_rate == 0.0

    def test_offset_within_tolerance(self):
        """Peaks within tolerance should still count as detected."""
        metric = FiducialDetectionMetric(tolerance=8)

        gt = np.zeros((9, 2400), dtype=np.float32)
        pred = np.zeros((9, 2400), dtype=np.float32)

        x = np.arange(2400, dtype=np.float32)
        gt[4] = np.exp(-0.5 * ((x - 200) / 4.0) ** 2)
        # Offset by 3 samples (within tolerance of 8)
        pred[4] = np.exp(-0.5 * ((x - 203) / 4.0) ** 2)

        metric.update(pred, gt)
        result = metric.compute()
        assert result.detection_rate["R_peak"] == pytest.approx(1.0)
        assert result.mae_samples["R_peak"] <= 8


class TestHeartRateMetric:
    def test_perfect_hr(self):
        """When predicted R-peaks match HR, MAE should be low."""
        metric = HeartRateMetric(fs=200)

        # 75 bpm = 0.8s intervals = 160 samples between peaks
        hm = np.zeros((9, 2400), dtype=np.float32)
        x = np.arange(2400, dtype=np.float32)
        for pos in range(100, 2300, 160):
            hm[4] += np.exp(-0.5 * ((x - pos) / 4.0) ** 2)

        metric.update(hm, gt_hr=75.0)
        result = metric.compute()
        assert result.mae_bpm < 5.0
        assert result.n_samples == 1

    def test_no_peaks(self):
        """When no peaks found, metric should still work."""
        metric = HeartRateMetric()
        hm = np.zeros((9, 2400), dtype=np.float32)
        metric.update(hm, gt_hr=75.0)
        result = metric.compute()
        assert result.n_samples == 0

    def test_reset(self):
        metric = HeartRateMetric()
        hm = np.zeros((9, 2400), dtype=np.float32)
        x = np.arange(2400, dtype=np.float32)
        for pos in range(100, 2300, 160):
            hm[4] += np.exp(-0.5 * ((x - pos) / 4.0) ** 2)
        metric.update(hm, gt_hr=75.0)
        metric.reset()
        result = metric.compute()
        assert result.n_samples == 0


class TestDenoiserMetric:
    def test_good_denoising(self):
        """When denoised is close to clean, SNR improvement should be positive."""
        metric = DenoiserMetric()

        rng = np.random.RandomState(42)
        clean = rng.randn(7, 2400).astype(np.float32)
        noisy = clean + 0.2 * rng.randn(7, 2400).astype(np.float32)
        # Near-perfect denoising (tiny residual)
        denoised = clean + 0.001 * rng.randn(7, 2400).astype(np.float32)

        metric.update(denoised, clean, noisy)
        result = metric.compute()

        assert result.snr_improvement_db > 0
        assert result.rmse < 0.01
        assert result.correlation > 0.99

    def test_no_improvement(self):
        """When denoised == noisy, SNR improvement should be ~0."""
        metric = DenoiserMetric()

        clean = np.random.randn(7, 2400).astype(np.float32)
        noisy = clean + 0.2 * np.random.randn(7, 2400).astype(np.float32)

        # No denoising at all
        metric.update(noisy, clean, noisy)
        result = metric.compute()
        assert abs(result.snr_improvement_db) < 0.5

    def test_reset(self):
        metric = DenoiserMetric()
        clean = np.random.randn(7, 2400).astype(np.float32)
        metric.update(clean, clean, clean + 0.1)
        metric.reset()
        result = metric.compute()
        assert result.n_samples == 0
