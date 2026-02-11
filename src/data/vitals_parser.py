"""Vitals data parser for alarm events."""

from __future__ import annotations

import json
from typing import Any, Optional

import h5py

from src.ecg_system.schemas import ThresholdViolation, VitalMeasurement, VitalsData


# Mapping from HDF5 vital group name to VitalsData field name
_VITAL_NAME_MAP: dict[str, str] = {
    "HR": "hr",
    "Pulse": "pulse",
    "SpO2": "spo2",
    "Systolic": "systolic",
    "Diastolic": "diastolic",
    "RespRate": "resp_rate",
    "Temp": "temp",
    "XL_Posture": "posture",
}


class VitalsParser:
    """
    Parse vital signs from alarm event.

    Supports: HR, Pulse, SpO2, Systolic, Diastolic, RespRate, Temp, XL_Posture.
    """

    VITAL_TYPES = list(_VITAL_NAME_MAP.keys())

    def parse_vital(
        self, vital_group: h5py.Group, vital_name: str
    ) -> VitalMeasurement:
        """Parse single vital measurement from its HDF5 group."""
        value = vital_group["value"][()]
        if hasattr(value, "item"):
            value = value.item()

        units = vital_group["units"][()]
        if isinstance(units, bytes):
            units = units.decode("utf-8")

        timestamp = float(vital_group["timestamp"][()])

        extras: dict[str, Any] = {}
        if "extras" in vital_group:
            raw = vital_group["extras"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                extras = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass

        return VitalMeasurement(
            name=vital_name,
            value=float(value),
            units=str(units),
            timestamp=timestamp,
            extras=extras,
        )

    def parse_all_vitals(self, vitals_group: h5py.Group) -> VitalsData:
        """Parse all vital measurements from event's vitals group."""
        vitals = VitalsData()
        for hdf5_name, field_name in _VITAL_NAME_MAP.items():
            if hdf5_name in vitals_group:
                measurement = self.parse_vital(vitals_group[hdf5_name], hdf5_name)
                setattr(vitals, field_name, measurement)
        return vitals

    def check_threshold_violations(
        self, vitals: VitalsData
    ) -> list[ThresholdViolation]:
        """Check if any vitals violate their thresholds."""
        violations: list[ThresholdViolation] = []
        for field_name in vitals.FIELD_NAMES:
            vital: Optional[VitalMeasurement] = getattr(vitals, field_name)
            if vital is None:
                continue
            if vital.is_above_threshold and vital.upper_threshold is not None:
                violations.append(
                    ThresholdViolation(
                        vital_name=vital.name,
                        value=vital.value,
                        units=vital.units,
                        threshold_type="upper",
                        threshold_value=vital.upper_threshold,
                    )
                )
            if vital.is_below_threshold and vital.lower_threshold is not None:
                violations.append(
                    ThresholdViolation(
                        vital_name=vital.name,
                        value=vital.value,
                        units=vital.units,
                        threshold_type="lower",
                        threshold_value=vital.lower_threshold,
                    )
                )
        return violations
