"""PyTorch Dataset for ECG alarm events."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import LoaderConfig
from src.data.hdf5_loader import HDF5AlarmEventLoader


class ECGAlarmDataset(Dataset):  # type: ignore[type-arg]
    """
    PyTorch Dataset for ECG alarm events from multiple HDF5 files.

    Each sample returns a dict with:
        ecg: Tensor[7, 2400]
        vitals: dict[str, float] (optional)
        ppg: Tensor[900] (optional)
        resp: Tensor[400] (optional)
        metadata: dict
        event_id: str
        patient_id: str
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        include_vitals: bool = True,
        include_auxiliary: bool = False,
        config: Optional[LoaderConfig] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.include_vitals = include_vitals
        self.include_auxiliary = include_auxiliary
        self._loader = HDF5AlarmEventLoader(config)

        # Index all events: list of (filepath, event_id, patient_id)
        self._index: list[tuple[str, str, str]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Scan all HDF5 files and build event index."""
        if not self.data_dir.exists():
            return
        for hdf5_path in sorted(self.data_dir.glob("*.h5")):
            try:
                with h5py.File(str(hdf5_path), "r") as f:
                    metadata = self._loader.load_metadata(f)
                    patient_id = metadata.patient_id
                    events = self._loader.list_events(f)
                    for event_id in events:
                        self._index.append((str(hdf5_path), event_id, patient_id))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        filepath, event_id, patient_id = self._index[idx]
        with h5py.File(filepath, "r") as f:
            event = self._loader.load_event(f, event_id)

        sample: dict[str, Any] = {
            "ecg": torch.from_numpy(event.ecg.as_array),
            "event_id": event.event_id,
            "patient_id": patient_id,
            "metadata": {
                "uuid": event.uuid,
                "timestamp": event.timestamp,
                "sample_rate": event.ecg.sample_rate,
                "num_samples": event.ecg.num_samples,
                "pacer_info": event.ecg.extras.get("pacer_info", 0),
            },
        }

        if self.include_vitals and event.vitals is not None:
            sample["vitals"] = event.vitals.to_dict()
        else:
            sample["vitals"] = {}

        if self.include_auxiliary:
            if event.ppg is not None:
                sample["ppg"] = torch.from_numpy(event.ppg.signal)
            if event.resp is not None:
                sample["resp"] = torch.from_numpy(event.resp.signal)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_event_by_id(
        self, patient_id: str, event_id: str
    ) -> Optional[dict[str, Any]]:
        """Load specific event by patient and event ID."""
        for idx, (fp, eid, pid) in enumerate(self._index):
            if pid == patient_id and eid == event_id:
                return self[idx]
        return None
