"""Unit tests for SignalQualityAssessor."""

import numpy as np
import pytest

from src.ecg_system.schemas import ECGData
from src.preprocessing.quality import SignalQualityAssessor

LEADS = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]
FS = 200
N = 2400


def _make_clean_ecg(seed: int = 42) -> ECGData:
    """Create a clean 7-lead ECG with realistic-ish morphology."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    signals: dict[str, np.ndarray] = {}
    for i, lead in enumerate(LEADS):
        # Simulated heartbeat: R-wave spikes + mild noise
        sig = np.zeros(N, dtype=np.float32)
        rr = 60.0 / 72.0  # 72 bpm
        for bt in np.arange(0, t[-1], rr):
            mask = np.abs(t - bt) < 0.02
            sig[mask] += 1.0 * np.exp(-((t[mask] - bt) ** 2) / (2 * 0.008**2))
        sig += rng.normal(0, 0.02, N).astype(np.float32)
        signals[lead] = sig
    return ECGData(
        signals=signals,
        sample_rate=FS,
        num_samples=N,
        duration_sec=12.0,
        extras={"pacer_info": 0, "pacer_offset": 0},
    )


def _make_ecg_with_flat_lead(flat_lead: str = "aVR") -> ECGData:
    """ECG where one lead is flat (lead-off)."""
    ecg = _make_clean_ecg()
    ecg.signals[flat_lead] = np.zeros(N, dtype=np.float32)
    return ecg


def _make_noisy_ecg() -> ECGData:
    """ECG where all leads are pure noise."""
    rng = np.random.default_rng(99)
    signals = {
        lead: rng.normal(0, 0.0001, N).astype(np.float32)
        for lead in LEADS
    }
    return ECGData(
        signals=signals,
        sample_rate=FS,
        num_samples=N,
        duration_sec=12.0,
        extras={"pacer_info": 0},
    )


class TestSignalQualityAssessor:

    def setup_method(self) -> None:
        self.assessor = SignalQualityAssessor()

    def test_clean_signal_high_sqi(self) -> None:
        """Clean ECG should have overall SQI >= 0.85 and all leads usable."""
        ecg = _make_clean_ecg()
        report = self.assessor.assess(ecg)

        assert report.overall_sqi >= 0.85, f"SQI too low: {report.overall_sqi}"
        assert len(report.usable_leads) == 7
        assert report.excluded_leads == []
        assert report.noise_level == "low"

    def test_clean_signal_no_flags(self) -> None:
        """Clean ECG should produce no quality flags."""
        ecg = _make_clean_ecg()
        report = self.assessor.assess(ecg)
        assert "signal_unusable" not in report.quality_flags
        assert "high_noise" not in report.quality_flags

    def test_lead_off_detection(self) -> None:
        """Flat lead should be excluded with SQI < 0.3 and lead_off flag."""
        ecg = _make_ecg_with_flat_lead("aVR")
        report = self.assessor.assess(ecg)

        assert report.lead_sqi["aVR"] < 0.3
        assert "aVR" in report.excluded_leads
        assert "aVR" not in report.usable_leads
        assert "lead_off_aVR" in report.quality_flags

    def test_pacer_detection(self) -> None:
        """Non-zero pacer_info should trigger pacer_detected flag."""
        ecg = _make_clean_ecg()
        ecg.extras["pacer_info"] = 1
        report = self.assessor.assess(ecg)
        assert "pacer_detected" in report.quality_flags

    def test_no_pacer_when_zero(self) -> None:
        """Zero pacer_info should not trigger pacer_detected."""
        ecg = _make_clean_ecg()
        report = self.assessor.assess(ecg)
        assert "pacer_detected" not in report.quality_flags

    def test_all_noise_signal_unusable(self) -> None:
        """All-noise signal should produce signal_unusable flag."""
        ecg = _make_noisy_ecg()
        report = self.assessor.assess(ecg)

        assert report.overall_sqi < 0.3
        assert "signal_unusable" in report.quality_flags

    def test_sqi_range(self) -> None:
        """All SQI values should be in [0, 1]."""
        ecg = _make_clean_ecg()
        report = self.assessor.assess(ecg)
        assert 0.0 <= report.overall_sqi <= 1.0
        for lead, sqi in report.lead_sqi.items():
            assert 0.0 <= sqi <= 1.0, f"{lead} SQI out of range: {sqi}"

    def test_noise_level_values(self) -> None:
        """Noise level should be one of the valid classifications."""
        ecg = _make_clean_ecg()
        report = self.assessor.assess(ecg)
        assert report.noise_level in {"low", "moderate", "high"}

    def test_baseline_stability_values(self) -> None:
        """Baseline stability should be one of the valid classifications."""
        ecg = _make_clean_ecg()
        report = self.assessor.assess(ecg)
        assert report.baseline_stability in {"stable", "moderate", "unstable"}
