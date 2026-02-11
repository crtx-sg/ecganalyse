"""Tests for the ECGSimulator facade."""

import numpy as np
import pytest

from src.simulator.conditions import Condition
from src.simulator.ecg_simulator import ECGSimulator, LEAD_NAMES, SimulatedEvent


class TestGenerateECG:
    def test_returns_7_leads(self):
        sim = ECGSimulator(seed=1)
        signals = sim.generate_ecg(Condition.NORMAL_SINUS)
        assert set(signals.keys()) == set(LEAD_NAMES)

    def test_lead_shape(self):
        sim = ECGSimulator(seed=1)
        signals = sim.generate_ecg(Condition.NORMAL_SINUS)
        for lead, sig in signals.items():
            assert sig.shape == (2400,), f"Lead {lead} wrong shape"
            assert sig.dtype == np.float32

    @pytest.mark.parametrize("cond", list(Condition))
    def test_all_conditions_generate(self, cond: Condition):
        sim = ECGSimulator(seed=42)
        signals = sim.generate_ecg(cond)
        assert len(signals) == 7
        for sig in signals.values():
            assert sig.shape == (2400,)

    def test_reproducible_with_seed(self):
        sig1 = ECGSimulator(seed=123).generate_ecg(Condition.NORMAL_SINUS)
        sig2 = ECGSimulator(seed=123).generate_ecg(Condition.NORMAL_SINUS)
        for lead in LEAD_NAMES:
            np.testing.assert_array_equal(sig1[lead], sig2[lead])

    def test_different_seeds_differ(self):
        sig1 = ECGSimulator(seed=1).generate_ecg(Condition.NORMAL_SINUS)
        sig2 = ECGSimulator(seed=2).generate_ecg(Condition.NORMAL_SINUS)
        assert not np.allclose(sig1["ECG1"], sig2["ECG1"])

    def test_noise_level_affects_signal(self):
        sim_clean = ECGSimulator(seed=10)
        sim_high = ECGSimulator(seed=10)
        clean = sim_clean.generate_ecg(Condition.NORMAL_SINUS, noise_level="clean")
        high = sim_high.generate_ecg(Condition.NORMAL_SINUS, noise_level="high")
        # Same seed but different noise means signals differ
        assert not np.allclose(clean["ECG1"], high["ECG1"])

    def test_custom_hr(self):
        sim = ECGSimulator(seed=5)
        signals = sim.generate_ecg(Condition.SINUS_TACHYCARDIA, hr=130.0)
        # Signal should not be flat
        assert np.std(signals["ECG2"]) > 0.01


class TestGenerateEvent:
    def test_returns_simulated_event(self):
        sim = ECGSimulator(seed=1)
        event = sim.generate_event(Condition.NORMAL_SINUS)
        assert isinstance(event, SimulatedEvent)

    def test_event_has_all_fields(self):
        sim = ECGSimulator(seed=1)
        event = sim.generate_event(Condition.ATRIAL_FIBRILLATION)
        assert event.condition == Condition.ATRIAL_FIBRILLATION
        assert event.hr > 0
        assert len(event.ecg_signals) == 7
        assert event.ppg_signal.shape[0] == 900
        assert event.resp_signal.shape[0] == int(12.0 * 33.33)  # 399
        assert len(event.vitals) == 8
        assert event.noise_level == "medium"

    def test_random_condition_selection(self):
        sim = ECGSimulator(seed=42)
        conditions_seen = set()
        for _ in range(50):
            event = sim.generate_event()
            conditions_seen.add(event.condition)
        # With 50 draws from 16 conditions, should see multiple
        assert len(conditions_seen) > 3

    def test_condition_proportions(self):
        sim = ECGSimulator(seed=0)
        proportions = {
            Condition.NORMAL_SINUS: 0.8,
            Condition.ATRIAL_FIBRILLATION: 0.2,
        }
        counts = {Condition.NORMAL_SINUS: 0, Condition.ATRIAL_FIBRILLATION: 0}
        for _ in range(100):
            event = sim.generate_event(condition_proportions=proportions)
            counts[event.condition] += 1
        # Normal should dominate
        assert counts[Condition.NORMAL_SINUS] > counts[Condition.ATRIAL_FIBRILLATION]

    def test_vitals_structure(self):
        sim = ECGSimulator(seed=1)
        event = sim.generate_event(Condition.NORMAL_SINUS)
        for key in ("HR", "Pulse", "SpO2", "Systolic", "Diastolic", "RespRate", "Temp", "XL_Posture"):
            assert key in event.vitals
            assert "value" in event.vitals[key]
            assert "units" in event.vitals[key]
            assert "timestamp" in event.vitals[key]

    def test_resp_signal_shape(self):
        """RESP samples = int(12 * 33.33) which is 399."""
        sim = ECGSimulator(seed=1)
        event = sim.generate_event(Condition.NORMAL_SINUS)
        expected = int(12.0 * 33.33)
        assert event.resp_signal.shape[0] == expected
