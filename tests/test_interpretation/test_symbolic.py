"""Unit tests for SymbolicCalculationEngine."""

import math

import numpy as np
import pytest

from src.ecg_system.schemas import Beat, FiducialPoint, GlobalMeasurements
from src.interpretation.symbolic import SymbolicCalculationEngine


def _make_beat(
    index: int,
    r_sample: int,
    fs: int = 200,
    has_p: bool = True,
    pr_ms: float = 160.0,
    qrs_ms: float = 85.0,
    qt_ms: float = 380.0,
) -> Beat:
    """Helper to build a Beat with fiducials and intervals."""
    fiducials = {
        "R_peak": FiducialPoint(
            sample=r_sample, time_ms=r_sample / fs * 1000.0, confidence=0.95,
        ),
    }
    if has_p:
        p_sample = r_sample - int(pr_ms / 1000.0 * fs)
        fiducials["P_peak"] = FiducialPoint(
            sample=max(0, p_sample), time_ms=max(0, p_sample) / fs * 1000.0, confidence=0.85,
        )

    intervals = {}
    if pr_ms:
        intervals["pr_interval_ms"] = pr_ms
    if qrs_ms:
        intervals["qrs_duration_ms"] = qrs_ms
    if qt_ms:
        intervals["qt_interval_ms"] = qt_ms

    return Beat(
        beat_index=index,
        beat_type="normal",
        lead="ECG2",
        fiducials=fiducials,
        intervals=intervals,
        morphology={},
        anomalies=[],
        anomaly_confidence=0.0,
    )


class TestComputeGlobalMeasurements:
    """Tests for compute_global_measurements."""

    def test_basic_measurements(self) -> None:
        """Correct HR and intervals from regular beats."""
        engine = SymbolicCalculationEngine(fs=200)
        # RR interval = 166 samples = 830ms → HR ~72.3 bpm
        beats = [_make_beat(i, r_sample=100 + i * 166) for i in range(5)]
        m = engine.compute_global_measurements(beats)

        assert isinstance(m, GlobalMeasurements)
        assert 70 < m.heart_rate_bpm < 75
        assert 825 < m.rr_mean_ms < 835
        assert m.rr_std_ms < 5  # very regular
        assert m.pr_interval_ms == 160.0
        assert m.qrs_duration_ms == 85.0
        assert m.qt_interval_ms == 380.0

    def test_no_beats(self) -> None:
        """Empty beats → zeros."""
        engine = SymbolicCalculationEngine(fs=200)
        m = engine.compute_global_measurements([])
        assert m.heart_rate_bpm == 0.0
        assert m.rr_mean_ms == 0.0

    def test_single_beat(self) -> None:
        """Single beat → no RR intervals → HR = 0."""
        engine = SymbolicCalculationEngine(fs=200)
        m = engine.compute_global_measurements([_make_beat(0, 200)])
        assert m.heart_rate_bpm == 0.0

    def test_qtc_bazett(self) -> None:
        """QTc Bazett = QT / sqrt(RR_sec)."""
        engine = SymbolicCalculationEngine(fs=200)
        beats = [_make_beat(i, r_sample=100 + i * 166, qt_ms=380) for i in range(5)]
        m = engine.compute_global_measurements(beats)

        rr_sec = m.rr_mean_ms / 1000.0
        expected = 380.0 / math.sqrt(rr_sec)
        assert abs(m.qtc_bazett_ms - expected) < 1.0

    def test_qtc_fridericia(self) -> None:
        """QTc Fridericia = QT / cbrt(RR_sec)."""
        engine = SymbolicCalculationEngine(fs=200)
        beats = [_make_beat(i, r_sample=100 + i * 166, qt_ms=380) for i in range(5)]
        m = engine.compute_global_measurements(beats)

        rr_sec = m.rr_mean_ms / 1000.0
        expected = 380.0 / (rr_sec ** (1.0 / 3.0))
        assert abs(m.qtc_fridericia_ms - expected) < 1.0

    def test_traces_populated(self) -> None:
        """Traces must be non-empty for a valid run."""
        engine = SymbolicCalculationEngine(fs=200)
        beats = [_make_beat(i, r_sample=100 + i * 166) for i in range(5)]
        engine.compute_global_measurements(beats)
        assert len(engine.traces) > 0
        assert any("Heart rate" in t for t in engine.traces)

    def test_variable_pr(self) -> None:
        """Median PR from beats with different PR intervals."""
        engine = SymbolicCalculationEngine(fs=200)
        beats = []
        pr_values = [150, 160, 170, 160, 155]
        for i, pr in enumerate(pr_values):
            beats.append(_make_beat(i, r_sample=100 + i * 166, pr_ms=float(pr)))
        m = engine.compute_global_measurements(beats)
        assert m.pr_interval_ms == 160.0  # median of [150,155,160,160,170]


class TestComputeRhythmMetrics:
    """Tests for compute_rhythm_metrics."""

    def test_all_p_waves(self) -> None:
        """All beats have P-waves → ratio = 1.0."""
        engine = SymbolicCalculationEngine(fs=200)
        beats = [_make_beat(i, r_sample=100 + i * 166, has_p=True) for i in range(5)]
        metrics = engine.compute_rhythm_metrics(beats)
        assert metrics["p_wave_presence_ratio"] == 1.0
        assert metrics["regularity_score"] > 0.9

    def test_no_p_waves(self) -> None:
        """No P-waves → ratio = 0."""
        engine = SymbolicCalculationEngine(fs=200)
        beats = [_make_beat(i, r_sample=100 + i * 166, has_p=False) for i in range(5)]
        metrics = engine.compute_rhythm_metrics(beats)
        assert metrics["p_wave_presence_ratio"] == 0.0

    def test_empty_beats(self) -> None:
        """No beats → zero metrics."""
        engine = SymbolicCalculationEngine(fs=200)
        metrics = engine.compute_rhythm_metrics([])
        assert metrics["p_wave_presence_ratio"] == 0.0
        assert metrics["regularity_score"] == 0.0

    def test_irregular_rhythm(self) -> None:
        """Irregular RR intervals → low regularity."""
        engine = SymbolicCalculationEngine(fs=200)
        # Very irregular: RR intervals vary wildly
        r_samples = [100, 300, 700, 800, 1400]
        beats = [_make_beat(i, r_sample=s) for i, s in enumerate(r_samples)]
        metrics = engine.compute_rhythm_metrics(beats)
        assert metrics["regularity_score"] < 0.8
