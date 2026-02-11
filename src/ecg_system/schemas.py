"""Data classes for ECG interpretation system input and output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ============== HDF5 Input Data Classes ==============


@dataclass
class FileMetadata:
    """Global metadata from HDF5 file."""

    patient_id: str
    sampling_rate_ecg: float  # 200.0 Hz
    sampling_rate_ppg: float  # 75.0 Hz
    sampling_rate_resp: float  # 33.33 Hz
    alarm_time_epoch: float
    alarm_offset_seconds: float  # 6.0
    seconds_before_event: float  # 6.0
    seconds_after_event: float  # 6.0
    data_quality_score: float  # 0.85-0.98
    device_info: str
    max_vital_history: int  # 30


@dataclass
class ECGData:
    """ECG signal data from alarm event."""

    signals: dict[str, np.ndarray]  # 7 leads
    sample_rate: int  # 200 Hz
    num_samples: int  # 2400
    duration_sec: float  # 12.0
    extras: dict[str, Any]  # pacer_info, etc.

    LEAD_ORDER: list[str] = field(
        default_factory=lambda: ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"],
        repr=False,
    )

    @property
    def leads(self) -> list[str]:
        return list(self.signals.keys())

    @property
    def as_array(self) -> np.ndarray:
        """Return as [7, 2400] array in standard lead order."""
        return np.stack([self.signals[lead] for lead in self.LEAD_ORDER])


@dataclass
class PPGData:
    """PPG signal data."""

    signal: np.ndarray  # [900 samples]
    sample_rate: float  # 75.0 Hz
    extras: dict[str, Any]


@dataclass
class RespData:
    """Respiratory signal data."""

    signal: np.ndarray  # [400 samples]
    sample_rate: float  # 33.33 Hz
    extras: dict[str, Any]


@dataclass
class VitalMeasurement:
    """Single vital sign measurement."""

    name: str
    value: float
    units: str
    timestamp: float
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def upper_threshold(self) -> Optional[float]:
        return self.extras.get("upper_threshold")

    @property
    def lower_threshold(self) -> Optional[float]:
        return self.extras.get("lower_threshold")

    @property
    def is_above_threshold(self) -> bool:
        if self.upper_threshold is not None:
            return self.value > self.upper_threshold
        return False

    @property
    def is_below_threshold(self) -> bool:
        if self.lower_threshold is not None:
            return self.value < self.lower_threshold
        return False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "value": self.value,
            "units": self.units,
            "timestamp": self.timestamp,
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
            "threshold_violation": self.is_above_threshold or self.is_below_threshold,
        }
        # Include posture-specific extras
        for key in ("step_count", "time_since_posture_change"):
            if key in self.extras:
                result[key] = self.extras[key]
        return result


@dataclass
class VitalsData:
    """All vital measurements from event."""

    hr: Optional[VitalMeasurement] = None
    pulse: Optional[VitalMeasurement] = None
    spo2: Optional[VitalMeasurement] = None
    systolic: Optional[VitalMeasurement] = None
    diastolic: Optional[VitalMeasurement] = None
    resp_rate: Optional[VitalMeasurement] = None
    temp: Optional[VitalMeasurement] = None
    posture: Optional[VitalMeasurement] = None

    FIELD_NAMES: list[str] = field(
        default_factory=lambda: [
            "hr", "pulse", "spo2", "systolic", "diastolic",
            "resp_rate", "temp", "posture",
        ],
        repr=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON assembly."""
        result: dict[str, Any] = {}
        for vital_name in self.FIELD_NAMES:
            vital = getattr(self, vital_name)
            if vital is not None:
                result[vital_name] = vital.to_dict()
        return result


@dataclass
class ThresholdViolation:
    """Record of a vital sign threshold violation."""

    vital_name: str
    value: float
    units: str
    threshold_type: str  # "upper" or "lower"
    threshold_value: float


@dataclass
class AlarmEvent:
    """Complete alarm event data."""

    event_id: str
    uuid: str
    timestamp: float
    ecg: ECGData
    ppg: Optional[PPGData] = None
    resp: Optional[RespData] = None
    vitals: Optional[VitalsData] = None
    metadata: Optional[FileMetadata] = None


# ============== Analysis Output Data Classes ==============


@dataclass
class FiducialPoint:
    """Fiducial point location with confidence."""

    sample: int
    time_ms: float
    confidence: float
    alternatives: Optional[list[FiducialPoint]] = None


@dataclass
class Beat:
    """Single beat analysis."""

    beat_index: int
    beat_type: str  # "normal", "pvc", "pac", "paced", "unclassified"
    lead: str
    fiducials: dict[str, FiducialPoint]
    intervals: dict[str, float]
    morphology: dict[str, Any]
    anomalies: list[str]
    anomaly_confidence: float


@dataclass
class GlobalMeasurements:
    """Global ECG measurements."""

    heart_rate_bpm: float
    rr_mean_ms: float
    rr_std_ms: float
    rr_min_ms: float
    rr_max_ms: float
    pr_interval_ms: Optional[float]
    pr_interval_range_ms: Optional[tuple[float, float]]
    qrs_duration_ms: float
    qrs_duration_range_ms: tuple[float, float]
    qt_interval_ms: float
    qtc_bazett_ms: float
    qtc_fridericia_ms: float


@dataclass
class RhythmAnalysis:
    """Rhythm classification results."""

    classification: str
    classification_confidence: float
    regularity: str
    regularity_score: float
    p_wave_morphology: str
    p_wave_presence_ratio: float
    p_qrs_relationship: str
    ectopic_beats: dict[str, Any]


@dataclass
class Finding:
    """Clinical finding with evidence."""

    finding_id: str
    finding: str
    category: str  # "rhythm", "conduction", "morphology", "ischemia", "vital", "validation"
    severity: str  # "normal", "mild", "moderate", "severe", "critical"
    confidence: float
    evidence: dict[str, Any]
    clinical_significance: Optional[str] = None


@dataclass
class QualityReport:
    """Signal quality assessment results."""

    overall_sqi: float
    lead_sqi: dict[str, float]
    usable_leads: list[str]
    excluded_leads: list[str]
    quality_flags: list[str]
    noise_level: str  # "low", "moderate", "high"
    baseline_stability: str  # "stable", "moderate", "unstable"
