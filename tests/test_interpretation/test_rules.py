"""Unit tests for RuleBasedReasoningEngine."""

import pytest

from src.ecg_system.schemas import (
    Beat,
    FiducialPoint,
    Finding,
    GlobalMeasurements,
    RhythmAnalysis,
)
from src.interpretation.rules import RuleBasedReasoningEngine


def _measurements(
    hr: float = 75.0,
    rr_std: float = 30.0,
    pr: float | None = 160.0,
    qrs: float = 85.0,
    qtc: float = 410.0,
) -> GlobalMeasurements:
    return GlobalMeasurements(
        heart_rate_bpm=hr,
        rr_mean_ms=60000 / hr if hr > 0 else 0,
        rr_std_ms=rr_std,
        rr_min_ms=700,
        rr_max_ms=900,
        pr_interval_ms=pr,
        pr_interval_range_ms=(pr - 5, pr + 5) if pr else None,
        qrs_duration_ms=qrs,
        qrs_duration_range_ms=(qrs - 3, qrs + 3),
        qt_interval_ms=380,
        qtc_bazett_ms=qtc,
        qtc_fridericia_ms=qtc - 10,
    )


def _make_beats(n: int = 10, beat_type: str = "normal") -> list[Beat]:
    beats = []
    for i in range(n):
        beats.append(Beat(
            beat_index=i,
            beat_type=beat_type,
            lead="ECG2",
            fiducials={
                "R_peak": FiducialPoint(sample=100 + i * 166, time_ms=0, confidence=0.9),
                "P_peak": FiducialPoint(sample=80 + i * 166, time_ms=0, confidence=0.8),
            },
            intervals={"qrs_duration_ms": 85},
            morphology={},
            anomalies=[],
            anomaly_confidence=0.0,
        ))
    return beats


class TestClassifyRhythm:
    """Tests for rhythm classification."""

    def test_normal_sinus_rhythm(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(hr=75)
        metrics = {"p_wave_presence_ratio": 0.95, "regularity_score": 0.95}
        rhythm = engine.classify_rhythm(m, metrics, _make_beats())

        assert isinstance(rhythm, RhythmAnalysis)
        assert rhythm.classification == "normal_sinus_rhythm"
        assert rhythm.classification_confidence >= 0.8

    def test_sinus_bradycardia(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(hr=50)
        metrics = {"p_wave_presence_ratio": 0.9, "regularity_score": 0.9}
        rhythm = engine.classify_rhythm(m, metrics, _make_beats())
        assert rhythm.classification == "sinus_bradycardia"

    def test_sinus_tachycardia(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(hr=110)
        metrics = {"p_wave_presence_ratio": 0.9, "regularity_score": 0.9}
        rhythm = engine.classify_rhythm(m, metrics, _make_beats())
        assert rhythm.classification == "sinus_tachycardia"

    def test_atrial_fibrillation(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(hr=85, rr_std=180)
        metrics = {"p_wave_presence_ratio": 0.2, "regularity_score": 0.4}
        rhythm = engine.classify_rhythm(m, metrics, _make_beats())
        assert rhythm.classification == "atrial_fibrillation"

    def test_zero_hr(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(hr=0)
        metrics = {"p_wave_presence_ratio": 0, "regularity_score": 0}
        rhythm = engine.classify_rhythm(m, metrics, [])
        assert rhythm.classification == "undetermined"
        assert rhythm.classification_confidence == 0.0

    def test_regularity_classification(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(hr=75)
        metrics = {"p_wave_presence_ratio": 0.95, "regularity_score": 0.95}
        rhythm = engine.classify_rhythm(m, metrics, _make_beats())
        assert rhythm.regularity == "regular"

    def test_p_wave_morphology(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(hr=75)
        # Low P ratio
        metrics = {"p_wave_presence_ratio": 0.1, "regularity_score": 0.95}
        rhythm = engine.classify_rhythm(m, metrics, _make_beats())
        assert rhythm.p_wave_morphology == "absent"

    def test_ectopic_counts(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(hr=75)
        metrics = {"p_wave_presence_ratio": 0.9, "regularity_score": 0.9}
        beats = _make_beats(10)
        beats[3] = Beat(
            beat_index=3, beat_type="pvc", lead="ECG2",
            fiducials=beats[3].fiducials, intervals={}, morphology={},
            anomalies=[], anomaly_confidence=0.0,
        )
        rhythm = engine.classify_rhythm(m, metrics, beats)
        assert rhythm.ectopic_beats["pvc_count"] == 1
        assert rhythm.ectopic_beats["total_beats"] == 10


class TestGenerateFindings:
    """Tests for finding generation."""

    def test_normal_findings(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements()
        metrics = {"p_wave_presence_ratio": 0.95, "regularity_score": 0.95}
        rhythm = engine.classify_rhythm(m, metrics, _make_beats())
        findings = engine.generate_findings(m, rhythm, _make_beats())

        assert isinstance(findings, list)
        assert len(findings) >= 1  # at least rhythm finding
        assert all(isinstance(f, Finding) for f in findings)
        # Rhythm finding is always first
        assert findings[0].category == "rhythm"

    def test_prolonged_pr(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(pr=250)
        metrics = {"p_wave_presence_ratio": 0.95, "regularity_score": 0.95}
        rhythm = engine.classify_rhythm(m, metrics, _make_beats())
        findings = engine.generate_findings(m, rhythm, _make_beats())

        conduction = [f for f in findings if f.category == "conduction"]
        assert any("Prolonged PR" in f.finding for f in conduction)

    def test_prolonged_qrs(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(qrs=140)
        metrics = {"p_wave_presence_ratio": 0.95, "regularity_score": 0.95}
        rhythm = engine.classify_rhythm(m, metrics, _make_beats())
        findings = engine.generate_findings(m, rhythm, _make_beats())

        conduction = [f for f in findings if f.category == "conduction"]
        assert any("Prolonged QRS" in f.finding for f in conduction)

    def test_severely_prolonged_qtc(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements(qtc=520)
        metrics = {"p_wave_presence_ratio": 0.95, "regularity_score": 0.95}
        rhythm = engine.classify_rhythm(m, metrics, _make_beats())
        findings = engine.generate_findings(m, rhythm, _make_beats())

        conduction = [f for f in findings if f.category == "conduction"]
        severe = [f for f in conduction if f.severity == "severe"]
        assert len(severe) >= 1
        assert any("Severely prolonged QTc" in f.finding for f in severe)

    def test_pvc_finding(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements()
        metrics = {"p_wave_presence_ratio": 0.9, "regularity_score": 0.9}
        beats = _make_beats(10)
        beats[2] = Beat(
            beat_index=2, beat_type="pvc", lead="ECG2",
            fiducials=beats[2].fiducials, intervals={}, morphology={},
            anomalies=[], anomaly_confidence=0.0,
        )
        rhythm = engine.classify_rhythm(m, metrics, beats)
        findings = engine.generate_findings(m, rhythm, beats)

        morph = [f for f in findings if f.category == "morphology"]
        assert any("PVC" in f.finding for f in morph)

    def test_wide_qrs_morphology(self) -> None:
        engine = RuleBasedReasoningEngine()
        m = _measurements()
        metrics = {"p_wave_presence_ratio": 0.9, "regularity_score": 0.9}
        beats = _make_beats(5)
        # Add wide QRS morphology flag
        for b in beats[:3]:
            b.morphology["qrs_wide"] = True
        rhythm = engine.classify_rhythm(m, metrics, beats)
        findings = engine.generate_findings(m, rhythm, beats)

        morph = [f for f in findings if f.category == "morphology"]
        assert any("Wide QRS" in f.finding for f in morph)
