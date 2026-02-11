"""Tests for the HDF5 writer — output must be readable by the Phase 0 loader."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.simulator.conditions import Condition
from src.simulator.ecg_simulator import ECGSimulator, LEAD_NAMES
from src.simulator.hdf5_writer import HDF5EventWriter


@pytest.fixture
def events():
    sim = ECGSimulator(seed=42)
    return [sim.generate_event(Condition.NORMAL_SINUS, hr=72.0) for _ in range(2)]


@pytest.fixture
def hdf5_path(tmp_path: Path, events):
    path = str(tmp_path / "test_writer.h5")
    HDF5EventWriter().write_file(path, events, patient_id="PT_WRITE")
    return path


class TestHDF5Structure:
    def test_file_readable(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            assert "metadata" in f
            assert "event_1001" in f
            assert "event_1002" in f

    def test_metadata_fields(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            meta = f["metadata"]
            assert float(meta["sampling_rate_ecg"][()]) == 200.0
            assert float(meta["sampling_rate_ppg"][()]) == 75.0
            pid = meta["patient_id"][()]
            if isinstance(pid, bytes):
                pid = pid.decode()
            assert pid == "PT_WRITE"
            assert "alarm_time_epoch" in meta
            assert "alarm_offset_seconds" in meta
            assert "seconds_before_event" in meta
            assert "seconds_after_event" in meta
            assert "data_quality_score" in meta
            assert "device_info" in meta
            assert "max_vital_history" in meta

    def test_ecg_leads_present(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            ecg = f["event_1001"]["ecg"]
            for lead in LEAD_NAMES:
                assert lead in ecg, f"Missing lead {lead}"
                assert ecg[lead].shape == (2400,)

    def test_ecg_extras_json(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            raw = f["event_1001"]["ecg"]["extras"][()]
            data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            assert "pacer_info" in data
            assert "pacer_offset" in data

    def test_ppg_present(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            ppg = f["event_1001"]["ppg"]
            assert "PPG" in ppg
            assert ppg["PPG"].shape[0] == 900
            assert "extras" in ppg

    def test_resp_present(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            resp = f["event_1001"]["resp"]
            assert "RESP" in resp
            assert "extras" in resp

    def test_vitals_structure(self, hdf5_path):
        expected_vitals = ["HR", "Pulse", "SpO2", "Systolic", "Diastolic", "RespRate", "Temp", "XL_Posture"]
        with h5py.File(hdf5_path, "r") as f:
            vitals = f["event_1001"]["vitals"]
            for vname in expected_vitals:
                assert vname in vitals, f"Missing vital {vname}"
                vg = vitals[vname]
                assert "value" in vg
                assert "units" in vg
                assert "timestamp" in vg
                assert "extras" in vg

    def test_event_timestamp_and_uuid(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            evt = f["event_1001"]
            assert "timestamp" in evt
            assert "uuid" in evt
            assert float(evt["timestamp"][()]) > 0

    def test_output_dir_created(self, tmp_path):
        """Writer should create parent directories."""
        path = str(tmp_path / "subdir" / "deep" / "test.h5")
        sim = ECGSimulator(seed=1)
        events = [sim.generate_event(Condition.NORMAL_SINUS)]
        HDF5EventWriter().write_file(path, events)
        assert Path(path).exists()
