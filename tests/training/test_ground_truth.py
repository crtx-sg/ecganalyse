"""Tests for ground truth fiducial extraction."""

import numpy as np
import pytest

from src.simulator.conditions import Condition, CONDITION_REGISTRY
from src.simulator.morphology import (
    BeatFiducials,
    PatientParams,
    generate_lead_with_fiducials,
    generate_patient_params,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def patient_params(rng):
    return generate_patient_params(rng)


class TestBeatFiducials:
    def test_dataclass_fields(self):
        fid = BeatFiducials()
        assert fid.p_onset is None
        assert fid.p_peak is None
        assert fid.p_offset is None
        assert fid.qrs_onset == 0
        assert fid.r_peak == 0
        assert fid.t_peak == 0

    def test_with_values(self):
        fid = BeatFiducials(
            p_onset=100, p_peak=110, p_offset=120,
            qrs_onset=140, r_peak=160, qrs_offset=180,
            t_onset=200, t_peak=220, t_offset=240,
        )
        assert fid.p_onset == 100
        assert fid.r_peak == 160
        assert fid.t_offset == 240


class TestGenerateLeadWithFiducials:
    def test_returns_signal_and_fiducials(self, rng, patient_params):
        fs = 200.0
        duration = 12.0
        time = np.linspace(0, duration, int(fs * duration), endpoint=False)
        cfg = CONDITION_REGISTRY[Condition.NORMAL_SINUS]

        signal, fiducials = generate_lead_with_fiducials(
            time, cfg, patient_params, hr=75.0, lead_scale=1.0, rng=rng,
        )

        assert isinstance(signal, np.ndarray)
        assert signal.shape == (2400,)
        assert isinstance(fiducials, list)
        assert len(fiducials) > 0
        assert all(isinstance(f, BeatFiducials) for f in fiducials)

    def test_fiducial_ordering(self, rng, patient_params):
        """Fiducial positions should follow P → QRS → T ordering."""
        time = np.linspace(0, 12.0, 2400, endpoint=False)
        cfg = CONDITION_REGISTRY[Condition.NORMAL_SINUS]

        _, fiducials = generate_lead_with_fiducials(
            time, cfg, patient_params, hr=75.0, lead_scale=1.0, rng=rng,
        )

        for fid in fiducials:
            if fid.p_onset is not None:
                assert fid.p_onset <= fid.p_peak
                assert fid.p_peak <= fid.p_offset
                assert fid.p_offset <= fid.qrs_onset
            assert fid.qrs_onset <= fid.r_peak
            assert fid.r_peak <= fid.qrs_offset
            assert fid.qrs_offset <= fid.t_onset
            assert fid.t_onset <= fid.t_peak
            assert fid.t_peak <= fid.t_offset

    def test_fiducials_in_range(self, rng, patient_params):
        """All fiducial positions should be within [0, n_samples-1]."""
        time = np.linspace(0, 12.0, 2400, endpoint=False)
        cfg = CONDITION_REGISTRY[Condition.NORMAL_SINUS]

        _, fiducials = generate_lead_with_fiducials(
            time, cfg, patient_params, hr=75.0, lead_scale=1.0, rng=rng,
        )

        for fid in fiducials:
            for attr in ('qrs_onset', 'r_peak', 'qrs_offset', 't_onset', 't_peak', 't_offset'):
                val = getattr(fid, attr)
                assert 0 <= val < 2400, f"{attr}={val} out of range"
            for attr in ('p_onset', 'p_peak', 'p_offset'):
                val = getattr(fid, attr)
                if val is not None:
                    assert 0 <= val < 2400, f"{attr}={val} out of range"

    def test_afib_missing_p_waves(self, rng, patient_params):
        """AFib has p_wave_presence=0.10, most beats should lack P waves."""
        time = np.linspace(0, 12.0, 2400, endpoint=False)
        cfg = CONDITION_REGISTRY[Condition.ATRIAL_FIBRILLATION]

        _, fiducials = generate_lead_with_fiducials(
            time, cfg, patient_params, hr=120.0, lead_scale=1.0, rng=rng,
        )

        p_present = sum(1 for f in fiducials if f.p_peak is not None)
        # Most should be missing
        assert p_present / len(fiducials) < 0.5

    def test_beat_count_matches_heart_rate(self, rng, patient_params):
        """Number of beats should approximately match the heart rate."""
        time = np.linspace(0, 12.0, 2400, endpoint=False)
        cfg = CONDITION_REGISTRY[Condition.NORMAL_SINUS]

        _, fiducials = generate_lead_with_fiducials(
            time, cfg, patient_params, hr=75.0, lead_scale=1.0, rng=rng,
        )

        # 75 bpm = ~15 beats in 12s. Allow some margin for edge effects.
        expected = 75.0 * 12.0 / 60.0
        assert abs(len(fiducials) - expected) <= 3

    @pytest.mark.parametrize("condition", [
        Condition.NORMAL_SINUS,
        Condition.SINUS_BRADYCARDIA,
        Condition.SINUS_TACHYCARDIA,
        Condition.LBBB,
        Condition.RBBB,
        Condition.PVC,
        Condition.AV_BLOCK_1,
        Condition.ST_ELEVATION,
    ])
    def test_all_conditions_produce_fiducials(self, condition, patient_params):
        """All conditions should produce at least one beat with fiducials."""
        rng = np.random.default_rng(123)
        time = np.linspace(0, 12.0, 2400, endpoint=False)
        cfg = CONDITION_REGISTRY[condition]
        hr = np.mean(cfg.hr_range)

        _, fiducials = generate_lead_with_fiducials(
            time, cfg, patient_params, hr=hr, lead_scale=1.0, rng=rng,
        )

        assert len(fiducials) > 0

    def test_signal_shape_matches_original(self, rng, patient_params):
        """Signal from generate_lead_with_fiducials should have same shape."""
        time = np.linspace(0, 12.0, 2400, endpoint=False)
        cfg = CONDITION_REGISTRY[Condition.NORMAL_SINUS]

        signal, _ = generate_lead_with_fiducials(
            time, cfg, patient_params, hr=75.0, lead_scale=1.0, rng=rng,
        )

        assert signal.shape == (2400,)
        assert signal.dtype == np.float64
