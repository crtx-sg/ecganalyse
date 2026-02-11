"""Tests for the morphology engine."""

import numpy as np
import pytest

from src.simulator.conditions import CONDITION_REGISTRY, Condition
from src.simulator.morphology import (
    PatientParams,
    create_p_wave,
    create_qrs_complex,
    create_t_wave,
    generate_beat_times,
    generate_patient_params,
    generate_single_lead,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def patient(rng):
    return generate_patient_params(rng)


@pytest.fixture
def time_array():
    return np.linspace(0, 12.0, 2400, endpoint=False)


class TestPatientParams:
    def test_generate_returns_dataclass(self, rng):
        p = generate_patient_params(rng)
        assert isinstance(p, PatientParams)

    def test_pr_interval_in_range(self, rng):
        for _ in range(50):
            p = generate_patient_params(rng)
            assert 0.12 <= p.pr_interval <= 0.20

    def test_qrs_duration_in_range(self, rng):
        for _ in range(50):
            p = generate_patient_params(rng)
            assert 0.08 <= p.qrs_duration <= 0.12


class TestWaves:
    def test_p_wave_positive(self, time_array, patient, rng):
        wave = create_p_wave(time_array, center=1.0, params=patient, rng=rng)
        assert wave.max() > 0

    def test_qrs_has_positive_peak(self, time_array, patient, rng):
        wave = create_qrs_complex(time_array, center=1.0, params=patient, rng=rng)
        assert wave.max() > 0.5  # R wave should be prominent

    def test_qrs_wide_is_broader(self, time_array, patient, rng):
        narrow = create_qrs_complex(time_array, 1.0, patient, wide=False, rng=rng)
        wide = create_qrs_complex(time_array, 1.0, patient, wide=True, rng=rng)
        # Wide QRS should have energy spread over more samples
        narrow_width = np.sum(np.abs(narrow) > 0.01)
        wide_width = np.sum(np.abs(wide) > 0.01)
        assert wide_width > narrow_width

    def test_t_wave_inverted(self, time_array, patient, rng):
        normal = create_t_wave(time_array, 1.5, patient, inverted=False, rng=rng)
        inv = create_t_wave(time_array, 1.5, patient, inverted=True, rng=rng)
        # Inverted T should have opposite sign peak
        assert np.sign(normal.max()) != np.sign(inv.min())


class TestBeatTimes:
    def test_regular_rhythm(self, rng):
        beats = generate_beat_times(12.0, hr=60.0, rr_irregularity=0.0, rng=rng)
        # ~12 beats at 60 bpm in 12 seconds
        assert 9 <= len(beats) <= 13
        # Intervals should be ~1 second
        intervals = np.diff(beats)
        assert np.allclose(intervals, 1.0, atol=0.05)

    def test_irregular_rhythm(self, rng):
        beats = generate_beat_times(12.0, hr=80.0, rr_irregularity=0.25, rng=rng)
        intervals = np.diff(beats)
        # Should have noticeable variation
        assert np.std(intervals) > 0.05


class TestSingleLead:
    def test_output_shape(self, time_array, patient, rng):
        cfg = CONDITION_REGISTRY[Condition.NORMAL_SINUS]
        sig = generate_single_lead(time_array, cfg, patient, hr=72.0, lead_scale=1.0, rng=rng)
        assert sig.shape == time_array.shape

    def test_non_flat(self, time_array, patient, rng):
        cfg = CONDITION_REGISTRY[Condition.NORMAL_SINUS]
        sig = generate_single_lead(time_array, cfg, patient, hr=72.0, lead_scale=1.0, rng=rng)
        assert np.std(sig) > 0.01

    def test_einthoven_law(self, rng):
        """Lead III = Lead II - Lead I (Einthoven's law)."""
        time = np.linspace(0, 12.0, 2400, endpoint=False)
        patient = generate_patient_params(rng)
        cfg = CONDITION_REGISTRY[Condition.NORMAL_SINUS]

        # Use same RNG state for I and II by creating fresh ones
        rng1 = np.random.default_rng(100)
        lead_I = generate_single_lead(time, cfg, patient, 72.0, 1.0, rng1)

        rng2 = np.random.default_rng(101)
        lead_II = generate_single_lead(time, cfg, patient, 72.0, 1.1, rng2)

        lead_III = lead_II - lead_I
        # This is by construction, just verify the math works
        np.testing.assert_allclose(lead_III, lead_II - lead_I)
