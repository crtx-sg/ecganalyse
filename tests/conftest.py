"""Shared pytest fixtures for ECG interpretation system tests."""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest


LEADS = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]


def _make_synthetic_ecg(num_samples: int = 2400, seed: int = 42) -> np.ndarray:
    """Generate simple synthetic ECG signal."""
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples) / 200.0
    signal = np.zeros(num_samples, dtype=np.float32)
    rr = 60.0 / 72.0
    for bt in np.arange(0, t[-1], rr):
        r_center = bt
        mask = np.abs(t - r_center) < 0.02
        signal[mask] += 1.0 * np.exp(-((t[mask] - r_center) ** 2) / (2 * 0.008**2))
    signal += rng.normal(0, 0.02, num_samples).astype(np.float32)
    return signal


@pytest.fixture
def sample_hdf5_path(tmp_path: Path) -> str:
    """Create a temporary sample HDF5 file and return its path."""
    filepath = str(tmp_path / "test_patient_2024-01.h5")
    base_ts = 1704537600.0

    with h5py.File(filepath, "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        meta.create_dataset("patient_id", data="PT_TEST")
        meta.create_dataset("sampling_rate_ecg", data=200.0)
        meta.create_dataset("sampling_rate_ppg", data=75.0)
        meta.create_dataset("sampling_rate_resp", data=33.33)
        meta.create_dataset("alarm_time_epoch", data=base_ts)
        meta.create_dataset("alarm_offset_seconds", data=6.0)
        meta.create_dataset("seconds_before_event", data=6.0)
        meta.create_dataset("seconds_after_event", data=6.0)
        meta.create_dataset("data_quality_score", data=0.94)
        meta.create_dataset("device_info", data="RMSAI-SimDevice-v1.0")
        meta.create_dataset("max_vital_history", data=30)

        for evt_idx in range(2):
            event_id = f"event_{1001 + evt_idx}"
            event = f.create_group(event_id)
            event_ts = base_ts + evt_idx * 60

            # ECG
            ecg = event.create_group("ecg")
            for i, lead in enumerate(LEADS):
                ecg.create_dataset(
                    lead,
                    data=_make_synthetic_ecg(seed=42 + i + evt_idx * 10),
                    compression="gzip",
                )
            ecg.create_dataset(
                "extras", data=json.dumps({"pacer_info": 0, "pacer_offset": 0})
            )

            # PPG
            ppg = event.create_group("ppg")
            ppg.create_dataset(
                "PPG",
                data=np.sin(2 * np.pi * 1.2 * np.arange(900) / 75.0).astype(np.float32),
                compression="gzip",
            )
            ppg.create_dataset("extras", data=json.dumps({}))

            # RESP
            resp = event.create_group("resp")
            resp.create_dataset(
                "RESP",
                data=np.sin(2 * np.pi * 0.25 * np.arange(400) / 33.33).astype(np.float32),
                compression="gzip",
            )
            resp.create_dataset("extras", data=json.dumps({}))

            # Vitals
            vitals = event.create_group("vitals")
            vitals_spec = {
                "HR": (73, "bpm", {"upper_threshold": 120, "lower_threshold": 50}),
                "Pulse": (74, "bpm", {"upper_threshold": 120, "lower_threshold": 50}),
                "SpO2": (96, "%", {"upper_threshold": 100, "lower_threshold": 90}),
                "Systolic": (125, "mmHg", {"upper_threshold": 160, "lower_threshold": 90}),
                "Diastolic": (78, "mmHg", {"upper_threshold": 100, "lower_threshold": 60}),
                "RespRate": (16, "brpm", {"upper_threshold": 30, "lower_threshold": 8}),
                "Temp": (37.1, "°C", {"upper_threshold": 38.5, "lower_threshold": 35.5}),
                "XL_Posture": (
                    45,
                    "degrees",
                    {"step_count": 1250, "time_since_posture_change": 120},
                ),
            }
            for vname, (val, units, extras) in vitals_spec.items():
                vg = vitals.create_group(vname)
                vg.create_dataset("value", data=val)
                vg.create_dataset("units", data=units)
                vg.create_dataset("timestamp", data=event_ts)
                vg.create_dataset("extras", data=json.dumps(extras))

            event.create_dataset("timestamp", data=event_ts)
            event.create_dataset("uuid", data=str(uuid.uuid4()))

    return filepath


@pytest.fixture
def sample_hdf5_no_vitals(tmp_path: Path) -> str:
    """HDF5 file with events but no vitals group."""
    filepath = str(tmp_path / "no_vitals.h5")
    with h5py.File(filepath, "w") as f:
        meta = f.create_group("metadata")
        meta.create_dataset("patient_id", data="PT_NOVITALS")
        meta.create_dataset("sampling_rate_ecg", data=200.0)
        meta.create_dataset("sampling_rate_ppg", data=75.0)
        meta.create_dataset("sampling_rate_resp", data=33.33)
        meta.create_dataset("alarm_time_epoch", data=1704537600.0)
        meta.create_dataset("alarm_offset_seconds", data=6.0)
        meta.create_dataset("seconds_before_event", data=6.0)
        meta.create_dataset("seconds_after_event", data=6.0)
        meta.create_dataset("data_quality_score", data=0.90)
        meta.create_dataset("device_info", data="RMSAI-SimDevice-v1.0")
        meta.create_dataset("max_vital_history", data=30)

        event = f.create_group("event_1001")
        ecg = event.create_group("ecg")
        for i, lead in enumerate(LEADS):
            ecg.create_dataset(
                lead, data=_make_synthetic_ecg(seed=42 + i), compression="gzip"
            )
        ecg.create_dataset("extras", data=json.dumps({"pacer_info": 0}))
        event.create_dataset("timestamp", data=1704537600.0)
        event.create_dataset("uuid", data=str(uuid.uuid4()))
    return filepath


@pytest.fixture
def sample_hdf5_no_aux(tmp_path: Path) -> str:
    """HDF5 file with events but no PPG/RESP groups."""
    filepath = str(tmp_path / "no_aux.h5")
    with h5py.File(filepath, "w") as f:
        meta = f.create_group("metadata")
        meta.create_dataset("patient_id", data="PT_NOAUX")
        meta.create_dataset("sampling_rate_ecg", data=200.0)
        meta.create_dataset("sampling_rate_ppg", data=75.0)
        meta.create_dataset("sampling_rate_resp", data=33.33)
        meta.create_dataset("alarm_time_epoch", data=1704537600.0)
        meta.create_dataset("alarm_offset_seconds", data=6.0)
        meta.create_dataset("seconds_before_event", data=6.0)
        meta.create_dataset("seconds_after_event", data=6.0)
        meta.create_dataset("data_quality_score", data=0.90)
        meta.create_dataset("device_info", data="RMSAI-SimDevice-v1.0")
        meta.create_dataset("max_vital_history", data=30)

        event = f.create_group("event_1001")
        ecg = event.create_group("ecg")
        for i, lead in enumerate(LEADS):
            ecg.create_dataset(
                lead, data=_make_synthetic_ecg(seed=42 + i), compression="gzip"
            )
        ecg.create_dataset("extras", data=json.dumps({"pacer_info": 0}))
        event.create_dataset("timestamp", data=1704537600.0)
        event.create_dataset("uuid", data=str(uuid.uuid4()))
    return filepath


@pytest.fixture
def empty_hdf5_path(tmp_path: Path) -> str:
    """HDF5 file with metadata but no events."""
    filepath = str(tmp_path / "empty.h5")
    with h5py.File(filepath, "w") as f:
        meta = f.create_group("metadata")
        meta.create_dataset("patient_id", data="PT_EMPTY")
        meta.create_dataset("sampling_rate_ecg", data=200.0)
        meta.create_dataset("sampling_rate_ppg", data=75.0)
        meta.create_dataset("sampling_rate_resp", data=33.33)
        meta.create_dataset("alarm_time_epoch", data=1704537600.0)
        meta.create_dataset("alarm_offset_seconds", data=6.0)
        meta.create_dataset("seconds_before_event", data=6.0)
        meta.create_dataset("seconds_after_event", data=6.0)
        meta.create_dataset("data_quality_score", data=0.90)
        meta.create_dataset("device_info", data="RMSAI-SimDevice-v1.0")
        meta.create_dataset("max_vital_history", data=30)
    return filepath
