"""Unit tests for VitalsParser."""

import h5py
import pytest

from src.data.vitals_parser import VitalsParser


class TestVitalsParser:
    def setup_method(self) -> None:
        self.parser = VitalsParser()

    def test_parse_all_vitals(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            vitals = self.parser.parse_all_vitals(f["event_1001"]["vitals"])

        assert vitals.hr is not None
        assert vitals.hr.name == "HR"
        assert vitals.hr.value == 73.0
        assert vitals.hr.units == "bpm"
        assert vitals.hr.timestamp > 0

        assert vitals.pulse is not None
        assert vitals.spo2 is not None
        assert vitals.systolic is not None
        assert vitals.diastolic is not None
        assert vitals.resp_rate is not None
        assert vitals.temp is not None
        assert vitals.posture is not None

    def test_parse_vital_thresholds(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            vitals = self.parser.parse_all_vitals(f["event_1001"]["vitals"])

        assert vitals.hr is not None
        assert vitals.hr.upper_threshold == 120
        assert vitals.hr.lower_threshold == 50

        assert vitals.spo2 is not None
        assert vitals.spo2.upper_threshold == 100
        assert vitals.spo2.lower_threshold == 90

    def test_threshold_not_violated(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            vitals = self.parser.parse_all_vitals(f["event_1001"]["vitals"])

        assert vitals.hr is not None
        assert not vitals.hr.is_above_threshold
        assert not vitals.hr.is_below_threshold

    def test_check_threshold_violations_none(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            vitals = self.parser.parse_all_vitals(f["event_1001"]["vitals"])
        violations = self.parser.check_threshold_violations(vitals)
        assert violations == []

    def test_threshold_violation_detection(self) -> None:
        """Test that threshold violations are correctly detected via schema."""
        from src.ecg_system.schemas import VitalMeasurement, VitalsData

        low_spo2 = VitalMeasurement(
            name="SpO2", value=85, units="%", timestamp=1.0,
            extras={"upper_threshold": 100, "lower_threshold": 90},
        )
        assert low_spo2.is_below_threshold is True
        assert low_spo2.is_above_threshold is False

        high_hr = VitalMeasurement(
            name="HR", value=130, units="bpm", timestamp=1.0,
            extras={"upper_threshold": 120, "lower_threshold": 50},
        )
        assert high_hr.is_above_threshold is True
        assert high_hr.is_below_threshold is False

        vitals = VitalsData(spo2=low_spo2, hr=high_hr)
        violations = self.parser.check_threshold_violations(vitals)
        assert len(violations) == 2
        names = {v.vital_name for v in violations}
        assert "SpO2" in names
        assert "HR" in names

    def test_vitals_to_dict(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            vitals = self.parser.parse_all_vitals(f["event_1001"]["vitals"])
        d = vitals.to_dict()
        assert "hr" in d
        assert d["hr"]["value"] == 73.0
        assert d["hr"]["units"] == "bpm"
        assert d["hr"]["threshold_violation"] is False
        assert "spo2" in d
        assert "posture" in d
        # Posture extras
        assert "step_count" in d["posture"]

    def test_parse_missing_vital_types(self, sample_hdf5_no_vitals: str) -> None:
        """File without vitals group — parser shouldn't be called, but empty VitalsData works."""
        from src.ecg_system.schemas import VitalsData
        vitals = VitalsData()
        assert vitals.hr is None
        assert vitals.to_dict() == {}
        violations = self.parser.check_threshold_violations(vitals)
        assert violations == []

    def test_posture_extras(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            vitals = self.parser.parse_all_vitals(f["event_1001"]["vitals"])
        assert vitals.posture is not None
        assert vitals.posture.value == 45.0
        assert vitals.posture.units == "degrees"
        assert vitals.posture.extras.get("step_count") == 1250
        assert vitals.posture.extras.get("time_since_posture_change") == 120
