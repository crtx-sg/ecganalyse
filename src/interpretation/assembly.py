"""JSON Feature Assembly builder for ECG interpretation output.

Assembles all interpretation results into a single JSON-serializable
document conforming to schema version 1.1.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

from config.settings import PipelineConfig
from src.ecg_system.schemas import (
    AlarmEvent,
    Beat,
    Finding,
    GlobalMeasurements,
    QualityReport,
    RhythmAnalysis,
    VitalsData,
)


class JSONAssembler:
    """Assemble complete JSON Feature Assembly v1.1.

    Collects outputs from all Phase 4 engines and builds the final
    JSON document.
    """

    SCHEMA_VERSION = "1.1"

    def assemble(
        self,
        event: AlarmEvent,
        quality: QualityReport,
        measurements: GlobalMeasurements | None,
        rhythm: RhythmAnalysis,
        beats: list[Beat],
        findings: list[Finding],
        calculation_traces: list[str],
        processing_start_time: float | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> dict[str, Any]:
        """Build the complete JSON Feature Assembly.

        Args:
            event: The original alarm event (provides context, vitals, metadata).
            quality: Signal quality report from Phase 1.
            measurements: Global measurements from SymbolicCalculationEngine.
                ``None`` when beat analysis is disabled.
            rhythm: Rhythm analysis from RuleBasedReasoningEngine.
            beats: Detected beats from Phase 3.
            findings: All clinical findings (rules + vitals).
            calculation_traces: Auditable trace strings.
            processing_start_time: time.monotonic() at pipeline start (optional).
            pipeline_config: Pipeline configuration controlling which sections
                are included in the output.

        Returns:
            JSON-serializable dictionary.
        """
        cfg = pipeline_config or PipelineConfig()
        now = datetime.now(timezone.utc)
        processing_ms = 0.0
        if processing_start_time is not None:
            processing_ms = (time.monotonic() - processing_start_time) * 1000.0

        result: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "generated_at": now.isoformat(),
            "processing_time_ms": round(processing_ms, 2),
            "event_context": self._build_event_context(event),
            "metadata": self._build_metadata(event),
            "vitals_context": self._build_vitals_context(event.vitals),
            "quality": self._build_quality(quality),
            "rhythm": self._build_rhythm(rhythm),
            "findings": self._build_findings(findings),
            "calculation_traces": calculation_traces,
            "pipeline_config": {
                "beat_analysis_enabled": cfg.enable_beat_analysis,
                "heart_rate_enabled": cfg.enable_heart_rate,
                "interval_measurements_enabled": cfg.enable_interval_measurements,
                "beat_detector": cfg.beat_detector,
            },
        }

        if cfg.enable_beat_analysis and measurements is not None:
            result["global_measurements"] = self._build_global_measurements(
                measurements, event.vitals,
            )
            result["beats"] = self._build_beats(beats)
        else:
            result["global_measurements"] = None
            result["beats"] = []

        result["summary"] = self._build_summary(rhythm, findings, measurements)

        return result

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_event_context(self, event: AlarmEvent) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "event_id": event.event_id,
            "event_uuid": event.uuid,
            "event_timestamp": event.timestamp,
        }
        if event.metadata is not None:
            ctx["patient_id"] = event.metadata.patient_id
            ctx["alarm_offset_seconds"] = event.metadata.alarm_offset_seconds
            ctx["device_info"] = event.metadata.device_info
        return ctx

    def _build_metadata(self, event: AlarmEvent) -> dict[str, Any]:
        ecg = event.ecg
        result: dict[str, Any] = {
            "strip_info": {
                "duration_sec": ecg.duration_sec,
                "num_samples": ecg.num_samples,
                "sample_rate_hz": ecg.sample_rate,
            },
            "leads": ecg.leads,
            "pacer_status": ecg.extras.get("pacer_info", 0) != 0,
        }
        return result

    def _build_vitals_context(
        self, vitals: VitalsData | None,
    ) -> dict[str, Any] | None:
        if vitals is None:
            return None
        return vitals.to_dict()

    def _build_quality(self, quality: QualityReport) -> dict[str, Any]:
        return {
            "overall_sqi": quality.overall_sqi,
            "lead_sqi": quality.lead_sqi,
            "usable_leads": quality.usable_leads,
            "excluded_leads": quality.excluded_leads,
            "quality_flags": quality.quality_flags,
            "noise_level": quality.noise_level,
            "baseline_stability": quality.baseline_stability,
        }

    def _build_global_measurements(
        self,
        m: GlobalMeasurements,
        vitals: VitalsData | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "heart_rate_bpm": m.heart_rate_bpm,
            "rr_intervals_ms": {
                "mean": m.rr_mean_ms,
                "std": m.rr_std_ms,
                "min": m.rr_min_ms,
                "max": m.rr_max_ms,
            },
            "qrs_duration_ms": {
                "median": m.qrs_duration_ms,
                "range_min": m.qrs_duration_range_ms[0],
                "range_max": m.qrs_duration_range_ms[1],
            },
            "qt_interval_ms": m.qt_interval_ms,
            "qtc_bazett_ms": m.qtc_bazett_ms,
            "qtc_fridericia_ms": m.qtc_fridericia_ms,
        }

        # PR interval (may be None)
        if m.pr_interval_ms is not None:
            pr_section: dict[str, Any] = {"median": m.pr_interval_ms}
            if m.pr_interval_range_ms is not None:
                pr_section["range_min"] = m.pr_interval_range_ms[0]
                pr_section["range_max"] = m.pr_interval_range_ms[1]
            result["pr_interval_ms"] = pr_section
        else:
            result["pr_interval_ms"] = None

        # HR validation
        if vitals is not None and vitals.hr is not None:
            monitor_hr = vitals.hr.value
            diff = abs(m.heart_rate_bpm - monitor_hr)
            if diff <= 5:
                status = "consistent"
            elif diff <= 20:
                status = "discrepant"
            else:
                status = "critical"
            result["heart_rate_validation"] = {
                "ecg_derived_hr": m.heart_rate_bpm,
                "monitor_hr": monitor_hr,
                "difference_bpm": round(diff, 1),
                "status": status,
            }

        return result

    def _build_rhythm(self, rhythm: RhythmAnalysis) -> dict[str, Any]:
        return {
            "classification": rhythm.classification,
            "classification_confidence": rhythm.classification_confidence,
            "regularity": rhythm.regularity,
            "regularity_score": rhythm.regularity_score,
            "p_wave_morphology": rhythm.p_wave_morphology,
            "p_wave_presence_ratio": rhythm.p_wave_presence_ratio,
            "p_qrs_relationship": rhythm.p_qrs_relationship,
            "ectopic_beats": rhythm.ectopic_beats,
        }

    def _build_beats(self, beats: list[Beat]) -> list[dict[str, Any]]:
        result = []
        for beat in beats:
            fid_dict: dict[str, Any] = {}
            for name, fp in beat.fiducials.items():
                fid_dict[name] = {
                    "sample": fp.sample,
                    "time_ms": fp.time_ms,
                    "confidence": round(fp.confidence, 4),
                }
            result.append({
                "beat_index": beat.beat_index,
                "beat_type": beat.beat_type,
                "lead": beat.lead,
                "fiducials": fid_dict,
                "intervals": beat.intervals,
                "morphology": beat.morphology,
                "anomalies": beat.anomalies,
                "anomaly_confidence": round(beat.anomaly_confidence, 4),
            })
        return result

    def _build_findings(self, findings: list[Finding]) -> list[dict[str, Any]]:
        result = []
        for f in findings:
            result.append({
                "finding_id": f.finding_id,
                "finding": f.finding,
                "category": f.category,
                "severity": f.severity,
                "confidence": f.confidence,
                "evidence": f.evidence,
                "clinical_significance": f.clinical_significance,
            })
        return result

    def _build_summary(
        self,
        rhythm: RhythmAnalysis,
        findings: list[Finding],
        measurements: GlobalMeasurements | None,
    ) -> dict[str, Any]:
        categories = sorted(set(f.category for f in findings))
        severity_rank = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3, "critical": 4}
        abnormal = [
            f for f in findings if severity_rank.get(f.severity, 0) >= 2
        ]
        critical = [
            f.finding for f in findings if f.severity == "critical"
        ]

        # Primary interpretation
        rhythm_label = rhythm.classification.replace("_", " ")
        if not abnormal:
            primary = (
                f"Normal sinus rhythm with normal conduction and repolarization."
                if rhythm.classification == "normal_sinus_rhythm"
                else f"{rhythm_label.capitalize()} detected."
            )
        else:
            abnormal_descs = [f.finding for f in abnormal[:3]]
            primary = f"{rhythm_label.capitalize()}. Notable: {'; '.join(abnormal_descs)}."

        return {
            "primary_interpretation": primary,
            "categories_present": categories,
            "abnormality_count": len(abnormal),
            "critical_findings": critical,
        }
