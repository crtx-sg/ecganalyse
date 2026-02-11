"""HDF5 writer that produces files compatible with the Phase 0 loader."""

from __future__ import annotations

import json
import random
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from src.simulator.ecg_simulator import (
    ECG_DURATION,
    FS_ECG,
    FS_PPG,
    FS_RESP,
    SimulatedEvent,
)


class HDF5EventWriter:
    """Write :class:`SimulatedEvent` objects to Phase-0-compatible HDF5 files."""

    def write_file(
        self,
        filepath: str,
        events: list[SimulatedEvent],
        patient_id: str | None = None,
    ) -> None:
        """Write a complete HDF5 file.

        Args:
            filepath: output path (should end in ``.h5``).
            events: list of simulated events.
            patient_id: patient identifier.  Generated if *None*.
        """
        if patient_id is None:
            patient_id = f"PT{random.randint(1000, 9999)}"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        event_timestamps = self._generate_event_timestamps(len(events))

        with h5py.File(filepath, "w") as hf:
            self._write_metadata(hf, patient_id, event_timestamps)
            for idx, event in enumerate(events):
                event_id = 1001 + idx
                self._write_event(hf, event_id, event, event_timestamps[idx])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_event_timestamps(n: int) -> list[datetime]:
        start = datetime.now() - timedelta(hours=random.uniform(1, 24))
        timestamps: list[datetime] = []
        current = start
        for _ in range(n):
            timestamps.append(current)
            current += timedelta(seconds=random.uniform(30, 300))
        return timestamps

    @staticmethod
    def _write_metadata(
        hf: h5py.File, patient_id: str, event_timestamps: list[datetime],
    ) -> None:
        meta = hf.create_group("metadata")
        meta.create_dataset("patient_id", data=np.bytes_(patient_id))
        meta.create_dataset("sampling_rate_ecg", data=FS_ECG)
        meta.create_dataset("sampling_rate_ppg", data=FS_PPG)
        meta.create_dataset("sampling_rate_resp", data=FS_RESP)

        alarm_time = event_timestamps[0]
        alarm_epoch = time.mktime(alarm_time.timetuple()) + alarm_time.microsecond / 1e6
        meta.create_dataset("alarm_time_epoch", data=alarm_epoch)
        meta.create_dataset("alarm_offset_seconds", data=ECG_DURATION / 2)
        meta.create_dataset("seconds_before_event", data=ECG_DURATION / 2)
        meta.create_dataset("seconds_after_event", data=ECG_DURATION / 2)
        meta.create_dataset("data_quality_score", data=random.uniform(0.85, 0.98))
        meta.create_dataset("device_info", data=np.bytes_("RMSAI-SimDevice-v2.0"))
        meta.create_dataset("max_vital_history", data=30)

    @staticmethod
    def _write_event(
        hf: h5py.File,
        event_id: int,
        event: SimulatedEvent,
        event_timestamp: datetime,
    ) -> None:
        grp = hf.create_group(f"event_{event_id}")

        # --- ECG ---
        ecg_grp = grp.create_group("ecg")
        for lead_name, signal in event.ecg_signals.items():
            ecg_grp.create_dataset(lead_name, data=signal, compression="gzip")
        ecg_extras = {
            "pacer_info": int(event.pacer_info),
            "pacer_offset": int(event.pacer_offset),
        }
        ecg_grp.create_dataset("extras", data=json.dumps(ecg_extras).encode("utf-8"))

        # --- PPG ---
        ppg_grp = grp.create_group("ppg")
        ppg_grp.create_dataset("PPG", data=event.ppg_signal, compression="gzip")
        ppg_grp.create_dataset("extras", data=json.dumps({}).encode("utf-8"))

        # --- RESP ---
        resp_grp = grp.create_group("resp")
        resp_grp.create_dataset("RESP", data=event.resp_signal, compression="gzip")
        resp_grp.create_dataset("extras", data=json.dumps({}).encode("utf-8"))

        # --- Vitals ---
        vitals_grp = grp.create_group("vitals")
        for vital_name, info in event.vitals.items():
            vg = vitals_grp.create_group(vital_name)
            vg.create_dataset("value", data=info["value"])
            units = info["units"]
            vg.create_dataset("units", data=units.encode("utf-8") if isinstance(units, str) else units)
            vg.create_dataset("timestamp", data=info["timestamp"])

            extras: dict = {}
            if vital_name != "XL_Posture":
                extras["upper_threshold"] = info.get("upper_threshold", 0)
                extras["lower_threshold"] = info.get("lower_threshold", 0)
            else:
                extras["step_count"] = info.get("step_count", 0)
                extras["time_since_posture_change"] = info.get("time_since_posture_change", 0)
            vg.create_dataset("extras", data=json.dumps(extras).encode("utf-8"))

        # --- Event-level metadata ---
        epoch = time.mktime(event_timestamp.timetuple()) + event_timestamp.microsecond / 1e6
        grp.create_dataset("timestamp", data=epoch)
        grp.create_dataset("uuid", data=str(uuid.uuid4()))

        grp.attrs["condition"] = event.condition.value
        grp.attrs["heart_rate"] = event.hr
        grp.attrs["event_timestamp"] = epoch
