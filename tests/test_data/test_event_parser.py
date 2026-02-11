"""Unit tests for AlarmEventParser."""

import json

import h5py
import numpy as np
import pytest

from src.data.event_parser import AlarmEventParser
from src.ecg_system.exceptions import EventParseError


class TestAlarmEventParser:
    def setup_method(self) -> None:
        self.parser = AlarmEventParser()

    def test_parse_ecg_7_leads(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            ecg = self.parser.parse_ecg(f["event_1001"], "event_1001")
        assert set(ecg.leads) == {"ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"}
        assert len(ecg.leads) == 7
        for lead in ecg.leads:
            assert ecg.signals[lead].shape == (2400,)
            assert ecg.signals[lead].dtype == np.float32
        assert ecg.sample_rate == 200
        assert ecg.duration_sec == 12.0

    def test_parse_ecg_as_array(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            ecg = self.parser.parse_ecg(f["event_1001"], "event_1001")
        arr = ecg.as_array
        assert arr.shape == (7, 2400)
        assert arr.dtype == np.float32

    def test_parse_ecg_extras(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            ecg = self.parser.parse_ecg(f["event_1001"], "event_1001")
        assert "pacer_info" in ecg.extras
        assert ecg.extras["pacer_info"] == 0

    def test_parse_ecg_missing_leads(self, tmp_path) -> None:
        """Event with missing ECG lead should raise EventParseError."""
        filepath = str(tmp_path / "bad.h5")
        with h5py.File(filepath, "w") as f:
            event = f.create_group("event_1001")
            ecg = event.create_group("ecg")
            # Only create 3 leads instead of 7
            for lead in ["ECG1", "ECG2", "ECG3"]:
                ecg.create_dataset(lead, data=np.zeros(2400, dtype=np.float32))
            ecg.create_dataset("extras", data=json.dumps({}))

        with h5py.File(filepath, "r") as f:
            with pytest.raises(EventParseError, match="Missing ECG leads"):
                self.parser.parse_ecg(f["event_1001"], "event_1001")

    def test_parse_ecg_missing_group(self, tmp_path) -> None:
        filepath = str(tmp_path / "no_ecg.h5")
        with h5py.File(filepath, "w") as f:
            f.create_group("event_1001")

        with h5py.File(filepath, "r") as f:
            with pytest.raises(EventParseError, match="Missing 'ecg' group"):
                self.parser.parse_ecg(f["event_1001"], "event_1001")

    def test_parse_ppg(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            ppg = self.parser.parse_ppg(f["event_1001"])
        assert ppg is not None
        assert ppg.signal.shape == (900,)
        assert ppg.sample_rate == 75.0

    def test_parse_ppg_missing(self, sample_hdf5_no_aux: str) -> None:
        with h5py.File(sample_hdf5_no_aux, "r") as f:
            ppg = self.parser.parse_ppg(f["event_1001"])
        assert ppg is None

    def test_parse_resp(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            resp = self.parser.parse_resp(f["event_1001"])
        assert resp is not None
        assert resp.signal.shape == (400,)
        assert resp.sample_rate == 33.33

    def test_parse_resp_missing(self, sample_hdf5_no_aux: str) -> None:
        with h5py.File(sample_hdf5_no_aux, "r") as f:
            resp = self.parser.parse_resp(f["event_1001"])
        assert resp is None

    def test_parse_full_event(self, sample_hdf5_path: str) -> None:
        with h5py.File(sample_hdf5_path, "r") as f:
            event = self.parser.parse_event(f["event_1001"], "event_1001")
        assert event.event_id == "event_1001"
        assert event.ecg is not None
        assert event.ppg is not None
        assert event.resp is not None
        assert event.vitals is not None
        assert event.uuid is not None
        assert event.timestamp > 0

    def test_parse_event_no_vitals(self, sample_hdf5_no_vitals: str) -> None:
        with h5py.File(sample_hdf5_no_vitals, "r") as f:
            event = self.parser.parse_event(f["event_1001"], "event_1001")
        assert event.ecg is not None
        assert event.vitals is None

    def test_parse_event_no_aux(self, sample_hdf5_no_aux: str) -> None:
        with h5py.File(sample_hdf5_no_aux, "r") as f:
            event = self.parser.parse_event(f["event_1001"], "event_1001")
        assert event.ecg is not None
        assert event.ppg is None
        assert event.resp is None
