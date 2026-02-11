"""Tests for the noise pipeline."""

import numpy as np
import pytest

from src.simulator.noise import (
    NOISE_PRESETS,
    NoiseConfig,
    add_baseline_wander,
    add_electrode_noise,
    add_emg_artifact,
    add_gaussian_noise,
    add_motion_artifact,
    add_powerline_interference,
    apply_noise_pipeline,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def clean_signal():
    return np.zeros(2400, dtype=np.float64)


@pytest.fixture
def time_array():
    return np.linspace(0, 12.0, 2400, endpoint=False)


class TestNoisePresets:
    def test_all_presets_exist(self):
        assert set(NOISE_PRESETS.keys()) == {"clean", "low", "medium", "high"}

    def test_clean_preset_all_zeros(self):
        c = NOISE_PRESETS["clean"]
        assert c.gaussian_std == 0.0
        assert c.baseline_wander_amp == 0.0
        assert c.emg_probability == 0.0
        assert c.motion_probability == 0.0
        assert c.powerline_probability == 0.0
        assert c.electrode_probability == 0.0


class TestIndividualStages:
    def test_baseline_wander_adds_low_freq(self, clean_signal, time_array, rng):
        cfg = NOISE_PRESETS["medium"]
        out = add_baseline_wander(clean_signal, time_array, rng, cfg)
        assert not np.allclose(out, clean_signal)
        # Baseline wander should be smooth (low frequency)
        diff = np.diff(out)
        assert np.max(np.abs(diff)) < 0.1

    def test_gaussian_noise_stats(self, clean_signal, rng):
        cfg = NoiseConfig(gaussian_std=0.10)
        out = add_gaussian_noise(clean_signal, rng, cfg)
        noise = out - clean_signal
        assert abs(np.mean(noise)) < 0.02
        assert abs(np.std(noise) - 0.10) < 0.02

    def test_emg_artifact_localised(self, clean_signal, rng):
        """EMG artifact should only affect a portion of the signal."""
        cfg = NoiseConfig(emg_probability=1.0)  # Force artifact
        out = add_emg_artifact(clean_signal, 200.0, rng, cfg)
        changed = np.abs(out - clean_signal) > 1e-10
        # Should not affect entire signal
        assert np.sum(changed) < len(clean_signal)

    def test_motion_artifact_spikes(self, clean_signal, rng):
        cfg = NoiseConfig(motion_probability=1.0)
        out = add_motion_artifact(clean_signal, rng, cfg)
        assert np.max(np.abs(out)) > 0.1

    def test_powerline_interference_periodic(self, clean_signal, time_array, rng):
        cfg = NoiseConfig(powerline_probability=1.0)
        out = add_powerline_interference(clean_signal, time_array, rng, cfg)
        assert not np.allclose(out, clean_signal)

    def test_electrode_noise_scales_down(self, rng):
        signal = np.ones(100)
        cfg = NoiseConfig(electrode_probability=1.0)
        out = add_electrode_noise(signal, rng, cfg)
        # Should be scaled down (factor 0.6–0.9)
        assert np.all(out < 1.0)
        assert np.all(out > 0.5)


class TestPipeline:
    def test_clean_produces_no_noise(self, clean_signal, time_array, rng):
        out = apply_noise_pipeline(clean_signal, time_array, 200.0, rng, NOISE_PRESETS["clean"])
        np.testing.assert_array_equal(out, clean_signal)

    def test_pipeline_changes_signal(self, time_array, rng):
        signal = np.sin(2 * np.pi * 1.0 * time_array)
        out = apply_noise_pipeline(signal, time_array, 200.0, rng, NOISE_PRESETS["high"])
        assert not np.allclose(out, signal)

    def test_higher_noise_more_distortion(self, time_array):
        signal = np.sin(2 * np.pi * 1.0 * time_array)
        rng_low = np.random.default_rng(99)
        rng_high = np.random.default_rng(99)
        out_low = apply_noise_pipeline(signal.copy(), time_array, 200.0, rng_low, NOISE_PRESETS["low"])
        out_high = apply_noise_pipeline(signal.copy(), time_array, 200.0, rng_high, NOISE_PRESETS["high"])
        # On average the high-noise version should differ more from the original
        err_low = np.mean(np.abs(out_low - signal))
        err_high = np.mean(np.abs(out_high - signal))
        assert err_high > err_low
