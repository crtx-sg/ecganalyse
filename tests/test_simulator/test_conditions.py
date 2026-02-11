"""Tests for the Condition enum and CONDITION_REGISTRY."""

import pytest

from src.simulator.conditions import Condition, ConditionConfig, CONDITION_REGISTRY


class TestConditionEnum:
    def test_all_16_conditions_defined(self):
        assert len(Condition) == 16

    def test_all_conditions_have_registry_entry(self):
        for cond in Condition:
            assert cond in CONDITION_REGISTRY, f"{cond.name} missing from registry"


class TestConditionConfig:
    @pytest.mark.parametrize("cond", list(Condition))
    def test_hr_range_valid(self, cond: Condition):
        cfg = CONDITION_REGISTRY[cond]
        lo, hi = cfg.hr_range
        assert lo > 0
        assert hi > lo

    @pytest.mark.parametrize("cond", list(Condition))
    def test_qrs_duration_range_valid(self, cond: Condition):
        cfg = CONDITION_REGISTRY[cond]
        lo, hi = cfg.qrs_duration_range
        assert lo >= 0.04
        assert hi <= 0.40
        assert hi >= lo

    @pytest.mark.parametrize("cond", list(Condition))
    def test_probabilities_in_range(self, cond: Condition):
        cfg = CONDITION_REGISTRY[cond]
        assert 0.0 <= cfg.p_wave_presence <= 1.0
        assert 0.0 <= cfg.t_wave_inversion_prob <= 1.0
        assert 0.0 <= cfg.rr_irregularity <= 1.0

    def test_wide_qrs_conditions(self):
        wide = [c for c in Condition if CONDITION_REGISTRY[c].wide_qrs]
        expected_wide = {
            Condition.PVC,
            Condition.VENTRICULAR_TACHYCARDIA,
            Condition.VENTRICULAR_FIBRILLATION,
            Condition.LBBB,
            Condition.RBBB,
        }
        assert set(wide) == expected_wide

    def test_st_elevation_condition(self):
        ste = [c for c in Condition if CONDITION_REGISTRY[c].st_elevation]
        assert ste == [Condition.ST_ELEVATION]
