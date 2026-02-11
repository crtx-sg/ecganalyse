"""Alarm event parser for extracting signals from HDF5 event groups."""

from __future__ import annotations

import json
from typing import Any, Optional

import h5py
import numpy as np

from config.settings import LoaderConfig
from src.ecg_system.exceptions import EventParseError
from src.ecg_system.schemas import (
    AlarmEvent,
    ECGData,
    FileMetadata,
    PPGData,
    RespData,
)
from src.data.vitals_parser import VitalsParser


class AlarmEventParser:
    """
    Parse individual alarm events from HDF5 structure.

    Extracts ECG (7 leads), PPG, RESP, vitals, and event metadata.
    """

    ECG_LEADS = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]

    def __init__(self, config: Optional[LoaderConfig] = None) -> None:
        self.config = config or LoaderConfig()
        self._vitals_parser = VitalsParser()

    def parse_ecg(self, event_group: h5py.Group, event_id: str) -> ECGData:
        """Extract 7-lead ECG data from event group."""
        if "ecg" not in event_group:
            raise EventParseError(event_id, "Missing 'ecg' group")

        ecg_group = event_group["ecg"]
        missing_leads = [lead for lead in self.ECG_LEADS if lead not in ecg_group]
        if missing_leads:
            raise EventParseError(
                event_id, f"Missing ECG leads: {missing_leads}"
            )

        signals: dict[str, np.ndarray] = {}
        for lead in self.ECG_LEADS:
            data = ecg_group[lead][()]
            signals[lead] = np.asarray(data, dtype=np.float32)

        extras = self._parse_extras(ecg_group)

        num_samples = signals[self.ECG_LEADS[0]].shape[0]
        return ECGData(
            signals=signals,
            sample_rate=self.config.ecg_sample_rate,
            num_samples=num_samples,
            duration_sec=num_samples / self.config.ecg_sample_rate,
            extras=extras,
        )

    def parse_ppg(self, event_group: h5py.Group) -> Optional[PPGData]:
        """Extract PPG signal. Returns None if not present."""
        if "ppg" not in event_group:
            return None
        ppg_group = event_group["ppg"]
        if "PPG" not in ppg_group:
            return None
        signal = np.asarray(ppg_group["PPG"][()], dtype=np.float32)
        extras = self._parse_extras(ppg_group)
        return PPGData(
            signal=signal,
            sample_rate=self.config.ppg_sample_rate,
            extras=extras,
        )

    def parse_resp(self, event_group: h5py.Group) -> Optional[RespData]:
        """Extract respiratory signal. Returns None if not present."""
        if "resp" not in event_group:
            return None
        resp_group = event_group["resp"]
        if "RESP" not in resp_group:
            return None
        signal = np.asarray(resp_group["RESP"][()], dtype=np.float32)
        extras = self._parse_extras(resp_group)
        return RespData(
            signal=signal,
            sample_rate=self.config.resp_sample_rate,
            extras=extras,
        )

    def parse_event(
        self,
        event_group: h5py.Group,
        event_id: str,
        metadata: Optional[FileMetadata] = None,
    ) -> AlarmEvent:
        """Parse complete alarm event."""
        ecg = self.parse_ecg(event_group, event_id)
        ppg = self.parse_ppg(event_group)
        resp = self.parse_resp(event_group)

        # Vitals (optional)
        vitals = None
        if "vitals" in event_group:
            vitals = self._vitals_parser.parse_all_vitals(event_group["vitals"])

        # Event metadata
        timestamp = float(event_group["timestamp"][()])
        event_uuid = event_group["uuid"][()]
        if isinstance(event_uuid, bytes):
            event_uuid = event_uuid.decode("utf-8")

        return AlarmEvent(
            event_id=event_id,
            uuid=str(event_uuid),
            timestamp=timestamp,
            ecg=ecg,
            ppg=ppg,
            resp=resp,
            vitals=vitals,
            metadata=metadata,
        )

    @staticmethod
    def _parse_extras(group: h5py.Group) -> dict[str, Any]:
        """Parse extras JSON field from a group."""
        if "extras" not in group:
            return {}
        raw = group["extras"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
