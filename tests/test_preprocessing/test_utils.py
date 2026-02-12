"""Unit tests for preprocessing utilities."""

import numpy as np
import pytest

from src.preprocessing.utils import bandpass_filter, remove_baseline_wander, normalize_leads

FS = 200  # Hz
N = 2400  # 12 seconds at 200 Hz


class TestBandpassFilter:
    """Tests for bandpass_filter()."""

    def test_attenuates_60hz(self) -> None:
        """60 Hz powerline noise should be attenuated by at least 20 dB."""
        t = np.arange(N) / FS
        # 10 Hz signal (in-band) + 60 Hz noise (out-of-band)
        sig_10hz = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        noise_60hz = 0.5 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
        raw = sig_10hz + noise_60hz

        filtered = bandpass_filter(raw, FS, low=0.5, high=40.0)

        # Measure 60 Hz power before and after
        fft_raw = np.fft.rfft(raw)
        fft_filt = np.fft.rfft(filtered)
        freqs = np.fft.rfftfreq(N, d=1.0 / FS)

        idx_60 = np.argmin(np.abs(freqs - 60.0))
        power_before = np.abs(fft_raw[idx_60]) ** 2
        power_after = np.abs(fft_filt[idx_60]) ** 2

        attenuation_db = 10 * np.log10(power_before / max(power_after, 1e-20))
        assert attenuation_db > 20, f"60 Hz attenuation only {attenuation_db:.1f} dB"

    def test_preserves_qrs_band(self) -> None:
        """Energy in the QRS frequency band (1-30 Hz) should be preserved."""
        t = np.arange(N) / FS
        # Multi-frequency signal in ECG band
        sig = (
            np.sin(2 * np.pi * 1.2 * t)
            + 0.8 * np.sin(2 * np.pi * 10 * t)
            + 0.3 * np.sin(2 * np.pi * 25 * t)
        ).astype(np.float32)

        filtered = bandpass_filter(sig, FS)

        # Energy ratio (in-band should be mostly preserved)
        energy_before = np.sum(sig**2)
        energy_after = np.sum(filtered**2)
        ratio = energy_after / energy_before
        assert ratio > 0.7, f"Too much energy lost: ratio={ratio:.3f}"

    def test_output_dtype_matches_input(self) -> None:
        rng = np.random.default_rng(0)
        sig = rng.normal(0, 1, N).astype(np.float32)
        out = bandpass_filter(sig, FS)
        assert out.dtype == np.float32

    def test_output_shape_matches_input(self) -> None:
        rng = np.random.default_rng(0)
        sig = rng.normal(0, 1, N).astype(np.float32)
        out = bandpass_filter(sig, FS)
        assert out.shape == sig.shape


class TestRemoveBaselineWander:
    """Tests for remove_baseline_wander()."""

    def test_removes_low_frequency_drift(self) -> None:
        t = np.arange(N) / FS
        # ECG-like + low-freq drift
        ecg = np.sin(2 * np.pi * 1.2 * t).astype(np.float32)
        drift = 0.5 * np.sin(2 * np.pi * 0.1 * t).astype(np.float32)
        raw = ecg + drift

        cleaned = remove_baseline_wander(raw, FS)

        # The drift component should be greatly reduced
        fft_cleaned = np.fft.rfft(cleaned)
        freqs = np.fft.rfftfreq(N, d=1.0 / FS)
        idx_01 = np.argmin(np.abs(freqs - 0.1))
        power_after = np.abs(fft_cleaned[idx_01]) ** 2
        fft_raw = np.fft.rfft(raw)
        power_before = np.abs(fft_raw[idx_01]) ** 2
        assert power_after < power_before * 0.1


class TestNormalizeLeads:
    """Tests for normalize_leads()."""

    def test_zero_mean_unit_var(self) -> None:
        rng = np.random.default_rng(42)
        ecg = rng.normal(5.0, 2.0, (7, N)).astype(np.float32)
        normed = normalize_leads(ecg)

        for i in range(7):
            assert abs(np.mean(normed[i])) < 0.01, f"Lead {i} mean not ~0"
            assert abs(np.std(normed[i]) - 1.0) < 0.01, f"Lead {i} std not ~1"

    def test_constant_signal_no_error(self) -> None:
        """Constant signal (std=0) should not cause division error."""
        ecg = np.ones((7, N), dtype=np.float32) * 3.0
        normed = normalize_leads(ecg)
        # Should be zero-mean, not NaN/Inf
        assert not np.any(np.isnan(normed))
        assert not np.any(np.isinf(normed))
        np.testing.assert_allclose(normed, 0.0, atol=1e-6)

    def test_output_shape_and_dtype(self) -> None:
        rng = np.random.default_rng(42)
        ecg = rng.normal(0, 1, (7, N)).astype(np.float32)
        normed = normalize_leads(ecg)
        assert normed.shape == (7, N)
        assert normed.dtype == np.float32
