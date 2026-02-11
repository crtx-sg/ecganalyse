"""HDF5 alarm event file loader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import h5py

from config.settings import LoaderConfig
from src.ecg_system.exceptions import HDF5LoadError, EventParseError
from src.ecg_system.schemas import AlarmEvent, FileMetadata
from src.data.event_parser import AlarmEventParser


class HDF5AlarmEventLoader:
    """
    Load and parse HDF5 clinical alarm event files.

    File structure:
        PatientID_YYYY-MM.h5
        ├── metadata/
        ├── event_1001/
        │   ├── ecg/ (7 leads × 2400 samples)
        │   ├── ppg/ (900 samples)
        │   ├── resp/ (400 samples)
        │   ├── vitals/ (HR, SpO2, BP, etc.)
        │   ├── timestamp
        │   └── uuid
        └── ...
    """

    EVENT_PATTERN = re.compile(r"^event_\d+$")

    def __init__(self, config: Optional[LoaderConfig] = None) -> None:
        self.config = config or LoaderConfig()
        self._parser = AlarmEventParser(self.config)

    def load_file(self, filepath: str) -> h5py.File:
        """Open HDF5 file. Caller is responsible for closing (use as context manager)."""
        path = Path(filepath)
        if not path.exists():
            raise HDF5LoadError(f"File not found: {filepath}")
        if not path.suffix == ".h5":
            raise HDF5LoadError(f"Expected .h5 file, got: {path.suffix}")
        try:
            return h5py.File(filepath, "r")
        except Exception as exc:
            raise HDF5LoadError(f"Cannot open HDF5 file '{filepath}': {exc}") from exc

    def list_events(self, hdf5_file: h5py.File) -> list[str]:
        """List all event IDs in file (event_1001, event_1002, ...)."""
        events = [
            key for key in hdf5_file.keys()
            if self.EVENT_PATTERN.match(key)
        ]
        return sorted(events)

    def load_event(self, hdf5_file: h5py.File, event_id: str) -> AlarmEvent:
        """Load complete alarm event data."""
        if event_id not in hdf5_file:
            available = self.list_events(hdf5_file)
            raise EventParseError(
                event_id,
                f"Event not found. Available events: {available}",
            )
        event_group = hdf5_file[event_id]
        metadata = self.load_metadata(hdf5_file)
        return self._parser.parse_event(event_group, event_id, metadata)

    def load_metadata(self, hdf5_file: h5py.File) -> FileMetadata:
        """Load global file metadata."""
        if "metadata" not in hdf5_file:
            raise HDF5LoadError("HDF5 file missing 'metadata' group")
        meta = hdf5_file["metadata"]

        def _read(name: str, default: object = None) -> object:
            if name not in meta:
                if default is not None:
                    return default
                raise HDF5LoadError(f"Missing metadata field: {name}")
            val = meta[name][()]
            if isinstance(val, bytes):
                return val.decode("utf-8")
            return val

        return FileMetadata(
            patient_id=str(_read("patient_id")),
            sampling_rate_ecg=float(_read("sampling_rate_ecg")),
            sampling_rate_ppg=float(_read("sampling_rate_ppg")),
            sampling_rate_resp=float(_read("sampling_rate_resp")),
            alarm_time_epoch=float(_read("alarm_time_epoch")),
            alarm_offset_seconds=float(_read("alarm_offset_seconds")),
            seconds_before_event=float(_read("seconds_before_event")),
            seconds_after_event=float(_read("seconds_after_event")),
            data_quality_score=float(_read("data_quality_score")),
            device_info=str(_read("device_info")),
            max_vital_history=int(_read("max_vital_history")),
        )
