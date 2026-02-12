"""Vitals context integrator for cross-modal clinical validation.

Compares ECG-derived heart rate against monitor HR, checks vital sign
thresholds, and generates cross-modal clinical findings.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

import yaml

from src.ecg_system.schemas import (
    Finding,
    GlobalMeasurements,
    VitalsData,
)

_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "rules_config.yaml"


def _load_rules(path: Path | None = None) -> dict:
    p = path or _DEFAULT_RULES_PATH
    with open(p) as f:
        return yaml.safe_load(f)


class VitalsContextIntegrator:
    """Integrate vital signs with ECG-derived measurements.

    Generates findings for HR validation, threshold violations,
    and cross-modal clinical correlations.

    Args:
        rules_path: Path to YAML rules config (default: config/rules_config.yaml).
    """

    def __init__(self, rules_path: Path | None = None) -> None:
        self.rules = _load_rules(rules_path)

    def integrate(
        self,
        measurements: GlobalMeasurements,
        vitals: VitalsData | None,
    ) -> list[Finding]:
        """Generate all vitals-related findings.

        Args:
            measurements: ECG-derived global measurements.
            vitals: Vital sign data from the alarm event (may be None).

        Returns:
            List of Finding objects with category="vital".
        """
        if vitals is None:
            return [Finding(
                finding_id=_uid(),
                finding="No vitals data available",
                category="vital",
                severity="mild",
                confidence=1.0,
                evidence={"vitals_available": False},
            )]

        findings: list[Finding] = []

        # HR validation (ECG vs monitor)
        findings.extend(self._validate_hr(measurements, vitals))

        # Threshold violation checks
        findings.extend(self._check_thresholds(vitals))

        # Cross-modal correlations
        findings.extend(self._cross_modal_findings(measurements, vitals))

        return findings

    # ------------------------------------------------------------------
    # HR Validation
    # ------------------------------------------------------------------

    def _validate_hr(
        self,
        measurements: GlobalMeasurements,
        vitals: VitalsData,
    ) -> list[Finding]:
        """Compare ECG-derived HR against monitor HR."""
        if vitals.hr is None:
            return []

        ecg_hr = measurements.heart_rate_bpm
        monitor_hr = vitals.hr.value
        diff = abs(ecg_hr - monitor_hr)

        vv = self.rules.get("vitals_validation", {})
        threshold = vv.get("hr_discrepancy_threshold_bpm", 5)
        severe = vv.get("hr_discrepancy_severe_bpm", 20)

        evidence = {
            "ecg_hr": ecg_hr,
            "monitor_hr": monitor_hr,
            "difference_bpm": round(diff, 1),
        }

        if diff <= threshold:
            return [Finding(
                finding_id=_uid(),
                finding="HR ECG-Monitor consistent",
                category="vital",
                severity="normal",
                confidence=0.99,
                evidence=evidence,
            )]
        elif diff > severe:
            return [Finding(
                finding_id=_uid(),
                finding=f"Critical HR discrepancy: ECG {ecg_hr:.0f} vs Monitor {monitor_hr:.0f} bpm (diff {diff:.0f})",
                category="vital",
                severity="critical",
                confidence=0.95,
                evidence=evidence,
                clinical_significance="Large ECG-monitor HR discrepancy may indicate artifact or arrhythmia",
            )]
        else:
            return [Finding(
                finding_id=_uid(),
                finding=f"HR discrepancy: ECG {ecg_hr:.0f} vs Monitor {monitor_hr:.0f} bpm (diff {diff:.0f})",
                category="vital",
                severity="moderate",
                confidence=0.90,
                evidence=evidence,
                clinical_significance="Moderate ECG-monitor HR discrepancy",
            )]

    # ------------------------------------------------------------------
    # Threshold Violations
    # ------------------------------------------------------------------

    def _check_thresholds(self, vitals: VitalsData) -> list[Finding]:
        """Check all vital signs against their configured thresholds."""
        findings: list[Finding] = []
        violations_found = False

        vitals_to_check = [
            ("hr", vitals.hr),
            ("pulse", vitals.pulse),
            ("spo2", vitals.spo2),
            ("systolic", vitals.systolic),
            ("diastolic", vitals.diastolic),
            ("resp_rate", vitals.resp_rate),
            ("temp", vitals.temp),
        ]

        for name, vital in vitals_to_check:
            if vital is None:
                continue

            if vital.is_above_threshold:
                violations_found = True
                findings.append(Finding(
                    finding_id=_uid(),
                    finding=f"{name} above threshold ({vital.value} {vital.units} > {vital.upper_threshold})",
                    category="vital",
                    severity=self._threshold_severity(name, "upper"),
                    confidence=0.95,
                    evidence={
                        "vital_name": name,
                        "value": vital.value,
                        "units": vital.units,
                        "threshold_type": "upper",
                        "threshold_value": vital.upper_threshold,
                    },
                ))

            if vital.is_below_threshold:
                violations_found = True
                findings.append(Finding(
                    finding_id=_uid(),
                    finding=f"{name} below threshold ({vital.value} {vital.units} < {vital.lower_threshold})",
                    category="vital",
                    severity=self._threshold_severity(name, "lower"),
                    confidence=0.95,
                    evidence={
                        "vital_name": name,
                        "value": vital.value,
                        "units": vital.units,
                        "threshold_type": "lower",
                        "threshold_value": vital.lower_threshold,
                    },
                ))

        if not violations_found:
            checked = [name for name, v in vitals_to_check if v is not None]
            findings.append(Finding(
                finding_id=_uid(),
                finding="All vitals within normal ranges",
                category="vital",
                severity="normal",
                confidence=1.0,
                evidence={
                    "vitals_checked": checked,
                    "all_within_thresholds": True,
                },
            ))

        return findings

    def _threshold_severity(self, vital_name: str, direction: str) -> str:
        """Map vital + direction to severity."""
        severe_vitals = {"spo2", "hr", "systolic"}
        if vital_name in severe_vitals:
            return "severe"
        return "moderate"

    # ------------------------------------------------------------------
    # Cross-Modal Findings
    # ------------------------------------------------------------------

    def _cross_modal_findings(
        self,
        measurements: GlobalMeasurements,
        vitals: VitalsData,
    ) -> list[Finding]:
        """Generate findings from correlating ECG with vitals."""
        findings: list[Finding] = []

        # Tachycardia + Hypertension → stress response
        if (
            measurements.heart_rate_bpm > 100
            and vitals.systolic is not None
            and vitals.systolic.value > 140
        ):
            findings.append(Finding(
                finding_id=_uid(),
                finding="Tachycardia with hypertension suggests stress response",
                category="vital",
                severity="moderate",
                confidence=0.7,
                evidence={
                    "heart_rate_bpm": measurements.heart_rate_bpm,
                    "systolic_mmhg": vitals.systolic.value,
                },
                clinical_significance="Combined tachycardia and hypertension may indicate sympathetic activation",
            ))

        # Bradycardia + Hypotension → hemodynamic concern
        if (
            measurements.heart_rate_bpm < 50
            and vitals.systolic is not None
            and vitals.systolic.value < 90
        ):
            findings.append(Finding(
                finding_id=_uid(),
                finding="Bradycardia with hypotension — hemodynamic concern",
                category="vital",
                severity="severe",
                confidence=0.8,
                evidence={
                    "heart_rate_bpm": measurements.heart_rate_bpm,
                    "systolic_mmhg": vitals.systolic.value,
                },
                clinical_significance="Bradycardia with hypotension may require urgent intervention",
            ))

        # Tachycardia + Low SpO2 → hypoxia-driven
        if (
            measurements.heart_rate_bpm > 100
            and vitals.spo2 is not None
            and vitals.spo2.value < 90
        ):
            findings.append(Finding(
                finding_id=_uid(),
                finding="Tachycardia with hypoxemia",
                category="vital",
                severity="severe",
                confidence=0.8,
                evidence={
                    "heart_rate_bpm": measurements.heart_rate_bpm,
                    "spo2_pct": vitals.spo2.value,
                },
                clinical_significance="Tachycardia may be compensatory for hypoxemia",
            ))

        return findings


def _uid() -> str:
    return f"F-{uuid.uuid4().hex[:8]}"
