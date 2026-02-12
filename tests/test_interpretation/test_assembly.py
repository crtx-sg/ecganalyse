"""Unit tests for JSONAssembler."""

import json
import time

import pytest

from src.ecg_system.schemas import (
    AlarmEvent,
    Beat,
    ECGData,
    FileMetadata,
    FiducialPoint,
    Finding,
    GlobalMeasurements,
    QualityReport,
    RhythmAnalysis,
    VitalMeasurement,
    VitalsData,
)
from src.interpretation.assembly import JSONAssembler

import numpy as np


def _make_event() -> AlarmEvent:
    signals = {lead: np.zeros(2400, dtype=np.float32) for lead in
               ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]}
    ecg = ECGData(
        signals=signals, sample_rate=200, num_samples=2400,
        duration_sec=12.0, extras={"pacer_info": 0},
    )
    vitals = VitalsData(
        hr=VitalMeasurement("hr", 73, "bpm", 1704537600.0,
                            {"upper_threshold": 120, "lower_threshold": 50}),
        spo2=VitalMeasurement("spo2", 96, "%", 1704537600.0,
                              {"upper_threshold": 100, "lower_threshold": 90}),
    )
    metadata = FileMetadata(
        patient_id="PT_TEST",
        sampling_rate_ecg=200.0,
        sampling_rate_ppg=75.0,
        sampling_rate_resp=33.33,
        alarm_time_epoch=1704537600.0,
        alarm_offset_seconds=6.0,
        seconds_before_event=6.0,
        seconds_after_event=6.0,
        data_quality_score=0.94,
        device_info="TestDevice-v1.0",
        max_vital_history=30,
    )
    return AlarmEvent(
        event_id="event_1001",
        uuid="test-uuid-1234",
        timestamp=1704537600.0,
        ecg=ecg,
        vitals=vitals,
        metadata=metadata,
    )


def _make_quality() -> QualityReport:
    return QualityReport(
        overall_sqi=0.92,
        lead_sqi={"ECG1": 0.95, "ECG2": 0.90, "ECG3": 0.93,
                   "aVR": 0.91, "aVL": 0.92, "aVF": 0.94, "vVX": 0.89},
        usable_leads=["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"],
        excluded_leads=[],
        quality_flags=[],
        noise_level="low",
        baseline_stability="stable",
    )


def _make_measurements() -> GlobalMeasurements:
    return GlobalMeasurements(
        heart_rate_bpm=72.1,
        rr_mean_ms=832.0,
        rr_std_ms=45.2,
        rr_min_ms=790.0,
        rr_max_ms=880.0,
        pr_interval_ms=160.0,
        pr_interval_range_ms=(155.0, 165.0),
        qrs_duration_ms=85.0,
        qrs_duration_range_ms=(82.0, 88.0),
        qt_interval_ms=380.0,
        qtc_bazett_ms=416.5,
        qtc_fridericia_ms=408.2,
    )


def _make_rhythm() -> RhythmAnalysis:
    return RhythmAnalysis(
        classification="normal_sinus_rhythm",
        classification_confidence=0.92,
        regularity="regular",
        regularity_score=0.98,
        p_wave_morphology="normal",
        p_wave_presence_ratio=0.95,
        p_qrs_relationship="1:1_consistent",
        ectopic_beats={"pvc_count": 0, "pac_count": 0, "total_beats": 10},
    )


def _make_beats(n: int = 3) -> list[Beat]:
    beats = []
    for i in range(n):
        beats.append(Beat(
            beat_index=i,
            beat_type="normal",
            lead="ECG2",
            fiducials={
                "R_peak": FiducialPoint(sample=100 + i * 166, time_ms=(100 + i * 166) / 200 * 1000, confidence=0.95),
            },
            intervals={"qrs_duration_ms": 85.0},
            morphology={},
            anomalies=[],
            anomaly_confidence=0.0,
        ))
    return beats


def _make_findings() -> list[Finding]:
    return [
        Finding(
            finding_id="F-test001",
            finding="Rhythm: normal sinus rhythm",
            category="rhythm",
            severity="normal",
            confidence=0.92,
            evidence={"classification": "normal_sinus_rhythm"},
        ),
        Finding(
            finding_id="F-test002",
            finding="HR ECG-Monitor consistent",
            category="vital",
            severity="normal",
            confidence=0.99,
            evidence={"ecg_hr": 72.1, "monitor_hr": 73},
        ),
    ]


class TestJSONAssembler:
    """Tests for JSONAssembler.assemble."""

    def test_schema_version(self) -> None:
        assembler = JSONAssembler()
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), ["trace1"],
        )
        assert result["schema_version"] == "1.1"

    def test_required_top_level_keys(self) -> None:
        assembler = JSONAssembler()
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), ["trace1"],
        )
        required = {
            "schema_version", "generated_at", "processing_time_ms",
            "event_context", "metadata", "vitals_context", "quality",
            "global_measurements", "rhythm", "beats", "findings",
            "calculation_traces", "summary",
        }
        assert required.issubset(result.keys())

    def test_json_serializable(self) -> None:
        assembler = JSONAssembler()
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), ["trace1"],
        )
        # Must not raise
        json_str = json.dumps(result, indent=2)
        assert len(json_str) > 100

    def test_event_context(self) -> None:
        assembler = JSONAssembler()
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), [],
        )
        ctx = result["event_context"]
        assert ctx["event_id"] == "event_1001"
        assert ctx["event_uuid"] == "test-uuid-1234"
        assert ctx["patient_id"] == "PT_TEST"

    def test_metadata_section(self) -> None:
        assembler = JSONAssembler()
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), [],
        )
        meta = result["metadata"]
        assert meta["strip_info"]["num_samples"] == 2400
        assert meta["strip_info"]["sample_rate_hz"] == 200
        assert meta["pacer_status"] is False

    def test_quality_section(self) -> None:
        assembler = JSONAssembler()
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), [],
        )
        q = result["quality"]
        assert q["overall_sqi"] == 0.92
        assert len(q["lead_sqi"]) == 7

    def test_hr_validation(self) -> None:
        assembler = JSONAssembler()
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), [],
        )
        gm = result["global_measurements"]
        assert "heart_rate_validation" in gm
        assert gm["heart_rate_validation"]["status"] == "consistent"

    def test_beats_preserved(self) -> None:
        assembler = JSONAssembler()
        beats = _make_beats(5)
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), beats, _make_findings(), [],
        )
        assert len(result["beats"]) == 5

    def test_findings_preserved(self) -> None:
        assembler = JSONAssembler()
        findings = _make_findings()
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), findings, [],
        )
        assert len(result["findings"]) == len(findings)

    def test_summary(self) -> None:
        assembler = JSONAssembler()
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), [],
        )
        summary = result["summary"]
        assert "primary_interpretation" in summary
        assert "categories_present" in summary
        assert "abnormality_count" in summary
        assert "critical_findings" in summary

    def test_processing_time(self) -> None:
        assembler = JSONAssembler()
        start = time.monotonic()
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), [],
            processing_start_time=start,
        )
        assert result["processing_time_ms"] >= 0

    def test_no_vitals(self) -> None:
        assembler = JSONAssembler()
        event = _make_event()
        event.vitals = None
        result = assembler.assemble(
            event, _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), [],
        )
        assert result["vitals_context"] is None
        assert "heart_rate_validation" not in result["global_measurements"]

    def test_traces_preserved(self) -> None:
        assembler = JSONAssembler()
        traces = ["RR mean=832ms", "HR=72.1 bpm"]
        result = assembler.assemble(
            _make_event(), _make_quality(), _make_measurements(),
            _make_rhythm(), _make_beats(), _make_findings(), traces,
        )
        assert result["calculation_traces"] == traces
