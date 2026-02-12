"""Symbolic calculation engine for deterministic ECG measurement extraction.

Computes clinical intervals and heart rate from fiducial points with
auditable calculation traces (IEC 62304 compliance).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.ecg_system.schemas import Beat, FiducialPoint, GlobalMeasurements


class SymbolicCalculationEngine:
    """Compute clinical ECG measurements from detected fiducial points.

    All calculations produce human-readable traces stored in ``self.traces``.

    Args:
        fs: Sampling rate in Hz (default 200).
    """

    def __init__(self, fs: int = 200) -> None:
        self.fs = fs
        self.traces: list[str] = []

    def compute_global_measurements(
        self, beats: list[Beat],
    ) -> GlobalMeasurements:
        """Compute global (aggregated) measurements across all beats.

        Args:
            beats: List of Beat objects from Phase 3.

        Returns:
            GlobalMeasurements dataclass.
        """
        self.traces.clear()

        # RR intervals from consecutive R-peaks
        rr_intervals = self._compute_rr_intervals(beats)

        # Heart rate
        hr = self._compute_heart_rate(rr_intervals)

        # RR statistics
        rr_mean = float(np.mean(rr_intervals)) if rr_intervals else 0.0
        rr_std = float(np.std(rr_intervals)) if rr_intervals else 0.0
        rr_min = float(np.min(rr_intervals)) if rr_intervals else 0.0
        rr_max = float(np.max(rr_intervals)) if rr_intervals else 0.0

        if rr_intervals:
            self.traces.append(
                f"RR intervals (n={len(rr_intervals)}): "
                f"mean={rr_mean:.1f}ms, std={rr_std:.1f}ms, "
                f"min={rr_min:.1f}ms, max={rr_max:.1f}ms"
            )

        # Per-beat intervals
        pr_values = self._collect_intervals(beats, "pr_interval_ms")
        qrs_values = self._collect_intervals(beats, "qrs_duration_ms")
        qt_values = self._collect_intervals(beats, "qt_interval_ms")

        # PR interval (median)
        pr_ms: Optional[float] = None
        pr_range: Optional[tuple[float, float]] = None
        if pr_values:
            pr_ms = float(np.median(pr_values))
            pr_range = (float(np.min(pr_values)), float(np.max(pr_values)))
            self.traces.append(
                f"PR interval: median={pr_ms:.1f}ms from {len(pr_values)} beats "
                f"(range {pr_range[0]:.1f}-{pr_range[1]:.1f}ms)"
            )

        # QRS duration (median)
        qrs_ms = float(np.median(qrs_values)) if qrs_values else 0.0
        qrs_range = (
            (float(np.min(qrs_values)), float(np.max(qrs_values)))
            if qrs_values
            else (0.0, 0.0)
        )
        if qrs_values:
            self.traces.append(
                f"QRS duration: median={qrs_ms:.1f}ms from {len(qrs_values)} beats "
                f"(range {qrs_range[0]:.1f}-{qrs_range[1]:.1f}ms)"
            )

        # QT interval (median)
        qt_ms = float(np.median(qt_values)) if qt_values else 0.0
        if qt_values:
            self.traces.append(
                f"QT interval: median={qt_ms:.1f}ms from {len(qt_values)} beats"
            )

        # QTc (Bazett and Fridericia)
        qtc_bazett = self._compute_qtc_bazett(qt_ms, rr_mean)
        qtc_fridericia = self._compute_qtc_fridericia(qt_ms, rr_mean)

        return GlobalMeasurements(
            heart_rate_bpm=round(hr, 1),
            rr_mean_ms=round(rr_mean, 1),
            rr_std_ms=round(rr_std, 1),
            rr_min_ms=round(rr_min, 1),
            rr_max_ms=round(rr_max, 1),
            pr_interval_ms=round(pr_ms, 1) if pr_ms is not None else None,
            pr_interval_range_ms=pr_range,
            qrs_duration_ms=round(qrs_ms, 1),
            qrs_duration_range_ms=qrs_range,
            qt_interval_ms=round(qt_ms, 1),
            qtc_bazett_ms=round(qtc_bazett, 1),
            qtc_fridericia_ms=round(qtc_fridericia, 1),
        )

    # ------------------------------------------------------------------
    # RR intervals and heart rate
    # ------------------------------------------------------------------

    def _compute_rr_intervals(self, beats: list[Beat]) -> list[float]:
        """Compute RR intervals in ms from consecutive R-peaks."""
        r_samples = []
        for beat in beats:
            if "R_peak" in beat.fiducials:
                r_samples.append(beat.fiducials["R_peak"].sample)

        if len(r_samples) < 2:
            self.traces.append("RR intervals: insufficient R-peaks (<2)")
            return []

        rr = []
        for i in range(1, len(r_samples)):
            interval_ms = (r_samples[i] - r_samples[i - 1]) / self.fs * 1000.0
            rr.append(interval_ms)
            self.traces.append(
                f"RR[{i}]: sample {r_samples[i-1]}→{r_samples[i]} = "
                f"{interval_ms:.1f}ms"
            )
        return rr

    def _compute_heart_rate(self, rr_intervals: list[float]) -> float:
        """Compute heart rate in BPM from RR intervals."""
        if not rr_intervals:
            self.traces.append("Heart rate: no RR intervals available, defaulting to 0")
            return 0.0
        mean_rr_ms = float(np.mean(rr_intervals))
        if mean_rr_ms <= 0:
            return 0.0
        hr = 60000.0 / mean_rr_ms
        self.traces.append(
            f"Heart rate: 60000 / {mean_rr_ms:.1f}ms = {hr:.1f} bpm"
        )
        return hr

    # ------------------------------------------------------------------
    # Per-beat interval collection
    # ------------------------------------------------------------------

    def _collect_intervals(
        self, beats: list[Beat], key: str,
    ) -> list[float]:
        """Collect a specific interval from all beats that have it."""
        values = []
        for beat in beats:
            if key in beat.intervals:
                values.append(beat.intervals[key])
        return values

    # ------------------------------------------------------------------
    # QTc calculations
    # ------------------------------------------------------------------

    def _compute_qtc_bazett(self, qt_ms: float, rr_mean_ms: float) -> float:
        """QTc Bazett = QT / sqrt(RR in seconds)."""
        if qt_ms <= 0 or rr_mean_ms <= 0:
            self.traces.append("QTc Bazett: insufficient data (QT or RR missing)")
            return 0.0
        rr_sec = rr_mean_ms / 1000.0
        qtc = qt_ms / math.sqrt(rr_sec)
        self.traces.append(
            f"QTc Bazett: {qt_ms:.1f} / sqrt({rr_sec:.3f}) = {qtc:.1f}ms"
        )
        return qtc

    def _compute_qtc_fridericia(self, qt_ms: float, rr_mean_ms: float) -> float:
        """QTc Fridericia = QT / cbrt(RR in seconds)."""
        if qt_ms <= 0 or rr_mean_ms <= 0:
            self.traces.append("QTc Fridericia: insufficient data (QT or RR missing)")
            return 0.0
        rr_sec = rr_mean_ms / 1000.0
        qtc = qt_ms / (rr_sec ** (1.0 / 3.0))
        self.traces.append(
            f"QTc Fridericia: {qt_ms:.1f} / cbrt({rr_sec:.3f}) = {qtc:.1f}ms"
        )
        return qtc

    # ------------------------------------------------------------------
    # Rhythm metrics
    # ------------------------------------------------------------------

    def compute_rhythm_metrics(
        self, beats: list[Beat],
    ) -> dict[str, float]:
        """Compute rhythm-level metrics for rule engine.

        Returns dict with: p_wave_presence_ratio, regularity_score.
        """
        total = len(beats)
        if total == 0:
            return {"p_wave_presence_ratio": 0.0, "regularity_score": 0.0}

        p_present = sum(1 for b in beats if "P_peak" in b.fiducials)
        p_ratio = p_present / total

        # Regularity from RR std
        rr = self._compute_rr_intervals(beats) if not hasattr(self, "_cached_rr") else []
        # Re-use traces already generated
        rr_intervals = []
        for i in range(1, len(beats)):
            if "R_peak" in beats[i].fiducials and "R_peak" in beats[i - 1].fiducials:
                rr_intervals.append(
                    (beats[i].fiducials["R_peak"].sample
                     - beats[i - 1].fiducials["R_peak"].sample)
                    / self.fs * 1000.0
                )
        if rr_intervals:
            cv = float(np.std(rr_intervals) / np.mean(rr_intervals))
            regularity = max(0.0, 1.0 - cv)
        else:
            regularity = 0.0

        return {
            "p_wave_presence_ratio": round(p_ratio, 3),
            "regularity_score": round(regularity, 3),
        }
