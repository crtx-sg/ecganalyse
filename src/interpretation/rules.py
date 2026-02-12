"""Rule-based reasoning engine for ECG clinical interpretation.

Applies deterministic clinical rules (loaded from ``config/rules_config.yaml``)
to measurements and beat morphology to generate findings.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

import yaml

from src.ecg_system.schemas import (
    Beat,
    Finding,
    GlobalMeasurements,
    RhythmAnalysis,
)

_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "rules_config.yaml"


def _load_rules(path: Path | None = None) -> dict:
    p = path or _DEFAULT_RULES_PATH
    with open(p) as f:
        return yaml.safe_load(f)


class RuleBasedReasoningEngine:
    """Apply clinical rules to ECG measurements and generate findings.

    Args:
        rules_path: Path to YAML rules config (default: config/rules_config.yaml).
    """

    def __init__(self, rules_path: Path | None = None) -> None:
        self.rules = _load_rules(rules_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_rhythm(
        self,
        measurements: GlobalMeasurements,
        rhythm_metrics: dict[str, float],
        beats: list[Beat],
    ) -> RhythmAnalysis:
        """Classify heart rhythm from measurements and beat data."""
        hr = measurements.heart_rate_bpm
        rr_std = measurements.rr_std_ms
        p_ratio = rhythm_metrics.get("p_wave_presence_ratio", 0.0)
        regularity = rhythm_metrics.get("regularity_score", 0.0)

        rhythm_rules = self.rules.get("rhythm", {})
        nsr = rhythm_rules.get("normal_sinus", {})

        # Classification logic
        classification = "undetermined"
        confidence = 0.5

        if hr <= 0:
            classification = "undetermined"
            confidence = 0.0
        elif hr < rhythm_rules.get("sinus_bradycardia", {}).get("hr_max_bpm", 60):
            classification = "sinus_bradycardia"
            confidence = 0.8
        elif hr >= rhythm_rules.get("sinus_tachycardia", {}).get("hr_min_bpm", 100):
            classification = "sinus_tachycardia"
            confidence = 0.8
        elif (
            nsr.get("hr_min_bpm", 60) <= hr <= nsr.get("hr_max_bpm", 100)
            and rr_std <= nsr.get("rr_std_max_ms", 120)
            and p_ratio >= nsr.get("p_wave_presence_min", 0.8)
        ):
            classification = "normal_sinus_rhythm"
            confidence = 0.9
        else:
            # Check for irregular rhythm (possible AFib)
            if rr_std > nsr.get("rr_std_max_ms", 120) and p_ratio < 0.5:
                classification = "atrial_fibrillation"
                confidence = 0.6
            elif p_ratio < nsr.get("p_wave_presence_min", 0.8):
                classification = "irregular_rhythm"
                confidence = 0.5
            else:
                classification = "normal_sinus_rhythm"
                confidence = 0.7

        # Regularity classification
        if regularity > 0.9:
            regularity_str = "regular"
        elif regularity > 0.7:
            regularity_str = "mostly_regular"
        else:
            regularity_str = "irregular"

        # P-wave morphology
        if p_ratio >= 0.8:
            p_morph = "normal"
        elif p_ratio >= 0.3:
            p_morph = "intermittent"
        else:
            p_morph = "absent"

        # P-QRS relationship
        pr = measurements.pr_interval_ms
        if p_ratio >= 0.8 and pr is not None:
            p_qrs = "1:1_consistent" if 120 <= pr <= 200 else "1:1_variable"
        elif p_ratio < 0.3:
            p_qrs = "dissociated"
        else:
            p_qrs = "variable"

        # Ectopic beat counts
        ectopic: dict[str, Any] = {
            "pvc_count": sum(1 for b in beats if b.beat_type == "pvc"),
            "pac_count": sum(1 for b in beats if b.beat_type == "pac"),
            "total_beats": len(beats),
        }

        return RhythmAnalysis(
            classification=classification,
            classification_confidence=round(confidence, 2),
            regularity=regularity_str,
            regularity_score=round(regularity, 3),
            p_wave_morphology=p_morph,
            p_wave_presence_ratio=round(p_ratio, 3),
            p_qrs_relationship=p_qrs,
            ectopic_beats=ectopic,
        )

    def generate_findings(
        self,
        measurements: GlobalMeasurements,
        rhythm: RhythmAnalysis,
        beats: list[Beat],
    ) -> list[Finding]:
        """Generate all clinical findings from measurements and rhythm."""
        findings: list[Finding] = []

        # Rhythm findings
        findings.append(self._rhythm_finding(rhythm))

        # Conduction findings
        findings.extend(self._conduction_findings(measurements))

        # Morphology findings from beats
        findings.extend(self._morphology_findings(beats))

        return findings

    # ------------------------------------------------------------------
    # Private finding generators
    # ------------------------------------------------------------------

    def _rhythm_finding(self, rhythm: RhythmAnalysis) -> Finding:
        """Generate rhythm finding."""
        clf = rhythm.classification
        severity_map = {
            "normal_sinus_rhythm": "normal",
            "sinus_bradycardia": "mild",
            "sinus_tachycardia": "mild",
            "atrial_fibrillation": "moderate",
            "irregular_rhythm": "mild",
            "undetermined": "mild",
        }
        severity = severity_map.get(clf, "mild")

        return Finding(
            finding_id=_uid(),
            finding=f"Rhythm: {clf.replace('_', ' ')}",
            category="rhythm",
            severity=severity,
            confidence=rhythm.classification_confidence,
            evidence={
                "classification": clf,
                "heart_rate_bpm": None,  # filled by caller if needed
                "regularity": rhythm.regularity,
                "p_wave_presence_ratio": rhythm.p_wave_presence_ratio,
            },
        )

    def _conduction_findings(
        self, m: GlobalMeasurements,
    ) -> list[Finding]:
        """Generate conduction findings (PR, QRS, QTc)."""
        findings: list[Finding] = []
        cond = self.rules.get("conduction", {})

        # Prolonged PR
        pr_threshold = cond.get("prolonged_pr", {}).get("threshold_ms", 200)
        if m.pr_interval_ms is not None and m.pr_interval_ms > pr_threshold:
            findings.append(Finding(
                finding_id=_uid(),
                finding=f"Prolonged PR interval ({m.pr_interval_ms:.0f}ms > {pr_threshold}ms)",
                category="conduction",
                severity="mild",
                confidence=0.85,
                evidence={"pr_interval_ms": m.pr_interval_ms, "threshold_ms": pr_threshold},
                clinical_significance="Possible first-degree AV block",
            ))

        # Prolonged QRS
        qrs_threshold = cond.get("prolonged_qrs", {}).get("threshold_ms", 120)
        if m.qrs_duration_ms > qrs_threshold:
            findings.append(Finding(
                finding_id=_uid(),
                finding=f"Prolonged QRS duration ({m.qrs_duration_ms:.0f}ms > {qrs_threshold}ms)",
                category="conduction",
                severity="moderate",
                confidence=0.80,
                evidence={"qrs_duration_ms": m.qrs_duration_ms, "threshold_ms": qrs_threshold},
                clinical_significance="Possible bundle branch block",
            ))

        # Prolonged QTc
        qtc_conf = cond.get("prolonged_qtc", {})
        severe_qtc = qtc_conf.get("severe_threshold_ms", 500)
        male_qtc = qtc_conf.get("male_threshold_ms", 450)
        if m.qtc_bazett_ms > severe_qtc:
            findings.append(Finding(
                finding_id=_uid(),
                finding=f"Severely prolonged QTc ({m.qtc_bazett_ms:.0f}ms > {severe_qtc}ms)",
                category="conduction",
                severity="severe",
                confidence=0.90,
                evidence={"qtc_bazett_ms": m.qtc_bazett_ms, "threshold_ms": severe_qtc},
                clinical_significance="High risk of torsades de pointes",
            ))
        elif m.qtc_bazett_ms > male_qtc:
            findings.append(Finding(
                finding_id=_uid(),
                finding=f"Prolonged QTc ({m.qtc_bazett_ms:.0f}ms > {male_qtc}ms)",
                category="conduction",
                severity="moderate",
                confidence=0.85,
                evidence={"qtc_bazett_ms": m.qtc_bazett_ms, "threshold_ms": male_qtc},
            ))

        return findings

    def _morphology_findings(self, beats: list[Beat]) -> list[Finding]:
        """Generate morphology findings from beat data."""
        findings: list[Finding] = []

        wide_qrs_count = sum(
            1 for b in beats if b.morphology.get("qrs_wide", False)
        )
        if wide_qrs_count > 0:
            ratio = wide_qrs_count / len(beats) if beats else 0
            findings.append(Finding(
                finding_id=_uid(),
                finding=f"Wide QRS complexes in {wide_qrs_count}/{len(beats)} beats",
                category="morphology",
                severity="moderate" if ratio > 0.5 else "mild",
                confidence=0.7,
                evidence={"wide_qrs_count": wide_qrs_count, "total_beats": len(beats)},
            ))

        pvc_count = sum(1 for b in beats if b.beat_type == "pvc")
        if pvc_count > 0:
            findings.append(Finding(
                finding_id=_uid(),
                finding=f"PVCs detected: {pvc_count} in {len(beats)} beats",
                category="morphology",
                severity="mild" if pvc_count <= 3 else "moderate",
                confidence=0.7,
                evidence={"pvc_count": pvc_count, "total_beats": len(beats)},
            ))

        return findings


def _uid() -> str:
    return f"F-{uuid.uuid4().hex[:8]}"
