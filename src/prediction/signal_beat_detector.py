"""Signal-based beat detector for direct ECG analysis.

Detects R-peaks and estimates fiducial points directly from raw ECG signals
using classical DSP methods. Useful as a fallback when neural heatmap models
are untrained, or for validation/comparison against the heatmap pipeline.

Usage:
    detector = SignalBasedBeatDetector(fs=200)
    beats = detector.detect(ecg_array)  # ecg_array: [7, 2400]
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks, butter, sosfiltfilt

from src.ecg_system.schemas import Beat, FiducialPoint

LEAD_ORDER = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]


class SignalBasedBeatDetector:
    """Detect beats and estimate fiducial points from raw ECG.

    Uses bandpass filtering + peak detection on the primary lead,
    then estimates P/QRS/T fiducials from waveform morphology.

    Args:
        fs: Sampling rate in Hz (default 200).
        primary_lead: Lead for R-peak detection (default "ECG2").
    """

    def __init__(self, fs: int = 200, primary_lead: str = "ECG2") -> None:
        self.fs = fs
        self.primary_lead = primary_lead

    def detect(self, ecg_array: np.ndarray) -> list[Beat]:
        """Detect beats from raw ECG.

        Args:
            ecg_array: [7, num_samples] numpy array in standard lead order.

        Returns:
            List of Beat objects with fiducials, intervals, and morphology.
        """
        lead_idx = LEAD_ORDER.index(self.primary_lead)
        signal = ecg_array[lead_idx].astype(np.float64)
        n_samples = len(signal)

        # Bandpass filter 1-40 Hz
        filtered = self._bandpass(signal, 1.0, 40.0)

        # Detect R-peaks
        r_peaks = self._detect_r_peaks(filtered)
        if len(r_peaks) == 0:
            return []

        # Segment beats around R-peaks
        boundaries = self._segment(r_peaks, n_samples)

        beats: list[Beat] = []
        for beat_idx, (start, end, r_sample) in enumerate(boundaries):
            fiducials = self._estimate_fiducials(signal, filtered, r_sample, start, end)
            fiducials = self._validate_ordering(fiducials)
            intervals = self._compute_intervals(fiducials)
            morphology = self._extract_morphology(signal, fiducials, start, end)
            beat_type, anomalies, anom_conf = self._classify_beat(fiducials, intervals, morphology)

            beats.append(Beat(
                beat_index=beat_idx,
                beat_type=beat_type,
                lead=self.primary_lead,
                fiducials=fiducials,
                intervals=intervals,
                morphology=morphology,
                anomalies=anomalies,
                anomaly_confidence=anom_conf,
            ))

        return beats

    # ------------------------------------------------------------------
    # Bandpass filter
    # ------------------------------------------------------------------

    def _bandpass(self, signal: np.ndarray, low: float, high: float) -> np.ndarray:
        sos = butter(4, [low, high], btype="band", fs=self.fs, output="sos")
        return sosfiltfilt(sos, signal)

    # ------------------------------------------------------------------
    # R-peak detection
    # ------------------------------------------------------------------

    def _detect_r_peaks(self, filtered: np.ndarray) -> np.ndarray:
        """Detect R-peaks using adaptive amplitude-based peak finding.

        Strategy: find all candidate peaks, then keep only the tall ones
        (R-peaks) by thresholding at 40% of the max peak height. This
        reliably separates R-peaks (~1.0 mV) from T-waves (~0.25 mV).
        """
        # Minimum RR distance: 300ms (200 bpm max)
        min_distance = int(0.3 * self.fs)

        # First pass: find all candidate peaks with low threshold
        candidates, props = find_peaks(
            filtered,
            height=0.05,
            distance=int(0.15 * self.fs),  # 150ms minimum between any peaks
        )

        if len(candidates) == 0:
            return candidates

        # Adaptive threshold: R-peaks are the tallest peaks
        heights = props["peak_heights"]
        max_height = np.max(heights)
        r_threshold = max_height * 0.4

        # Keep only peaks above the R-peak threshold
        r_mask = heights > r_threshold
        r_candidates = candidates[r_mask]

        # Re-filter with minimum RR distance
        if len(r_candidates) < 2:
            return r_candidates

        # Enforce minimum distance between R-peaks
        kept = [r_candidates[0]]
        for i in range(1, len(r_candidates)):
            if r_candidates[i] - kept[-1] >= min_distance:
                kept.append(r_candidates[i])
        return np.array(kept)

    # ------------------------------------------------------------------
    # Beat segmentation
    # ------------------------------------------------------------------

    def _segment(
        self, r_peaks: np.ndarray, n_samples: int,
    ) -> list[tuple[int, int, int]]:
        boundaries: list[tuple[int, int, int]] = []
        for i, r in enumerate(r_peaks):
            start = 0 if i == 0 else (r_peaks[i - 1] + r) // 2
            end = n_samples if i == len(r_peaks) - 1 else (r + r_peaks[i + 1]) // 2
            boundaries.append((int(start), int(end), int(r)))
        return boundaries

    # ------------------------------------------------------------------
    # Fiducial estimation from waveform
    # ------------------------------------------------------------------

    def _estimate_fiducials(
        self,
        raw: np.ndarray,
        filtered: np.ndarray,
        r_sample: int,
        start: int,
        end: int,
    ) -> dict[str, FiducialPoint]:
        """Estimate all 9 fiducial points from signal morphology."""
        fiducials: dict[str, FiducialPoint] = {}
        n = len(raw)

        # R-peak (known)
        fiducials["R_peak"] = self._fp(r_sample, confidence=0.95)

        # QRS onset: search backwards from R-peak for zero-crossing or slope change
        # Typically 30-60ms before R-peak
        qrs_search_start = max(start, r_sample - int(0.08 * self.fs))
        qrs_search_end = r_sample
        qrs_onset = self._find_onset(filtered, qrs_search_start, qrs_search_end, direction="backward")
        if qrs_onset is not None:
            fiducials["QRS_onset"] = self._fp(qrs_onset, confidence=0.85)

        # QRS offset: search forward from R-peak
        qrs_off_start = r_sample
        qrs_off_end = min(end, r_sample + int(0.08 * self.fs))
        qrs_offset = self._find_onset(filtered, qrs_off_start, qrs_off_end, direction="forward")
        if qrs_offset is not None:
            fiducials["QRS_offset"] = self._fp(qrs_offset, confidence=0.85)

        # P-wave: search 120-250ms before R-peak
        p_region_start = max(start, r_sample - int(0.25 * self.fs))
        p_region_end = max(start, r_sample - int(0.08 * self.fs))
        if p_region_end > p_region_start + 5:
            p_peak = self._find_wave_peak(filtered, p_region_start, p_region_end)
            if p_peak is not None:
                fiducials["P_peak"] = self._fp(p_peak, confidence=0.80)
                # P onset: ~40ms before P_peak
                p_on = max(start, p_peak - int(0.04 * self.fs))
                fiducials["P_onset"] = self._fp(p_on, confidence=0.70)
                # P offset: ~40ms after P_peak
                p_off = min(p_region_end, p_peak + int(0.04 * self.fs))
                fiducials["P_offset"] = self._fp(p_off, confidence=0.70)

        # T-wave: search 150-400ms after R-peak
        t_region_start = min(end - 1, r_sample + int(0.15 * self.fs))
        t_region_end = min(end, r_sample + int(0.40 * self.fs))
        if t_region_end > t_region_start + 5:
            t_peak = self._find_wave_peak(filtered, t_region_start, t_region_end)
            if t_peak is not None:
                fiducials["T_peak"] = self._fp(t_peak, confidence=0.75)
                # T onset: ~60ms before T_peak
                t_on = max(t_region_start, t_peak - int(0.06 * self.fs))
                fiducials["T_onset"] = self._fp(t_on, confidence=0.65)
                # T offset: ~80ms after T_peak
                t_off = min(end - 1, t_peak + int(0.08 * self.fs))
                fiducials["T_offset"] = self._fp(t_off, confidence=0.65)

        return fiducials

    def _fp(self, sample: int, confidence: float) -> FiducialPoint:
        return FiducialPoint(
            sample=int(sample),
            time_ms=round(sample / self.fs * 1000.0, 2),
            confidence=confidence,
        )

    def _find_onset(
        self, sig: np.ndarray, start: int, end: int, direction: str,
    ) -> int | None:
        """Find QRS onset/offset by looking for where signal crosses baseline."""
        if end <= start:
            return None
        segment = sig[start:end]
        baseline = np.median(segment)
        threshold = baseline + 0.15 * (np.max(np.abs(segment - baseline)))

        if direction == "backward":
            # Walk backward from end
            for i in range(len(segment) - 1, 0, -1):
                if abs(segment[i] - baseline) < threshold:
                    return start + i
        else:
            # Walk forward from start
            for i in range(len(segment)):
                if abs(segment[i] - baseline) < threshold:
                    return start + i
        return start if direction == "backward" else end - 1

    def _find_wave_peak(
        self, sig: np.ndarray, start: int, end: int,
    ) -> int | None:
        """Find the dominant peak in a search region."""
        if end <= start + 2:
            return None
        segment = sig[start:end]
        # Look for positive peak first
        peaks, props = find_peaks(segment, prominence=0.01 * np.std(segment))
        if len(peaks) > 0:
            best = peaks[np.argmax(segment[peaks])]
            return start + int(best)
        # Fallback: argmax
        idx = int(np.argmax(np.abs(segment)))
        if np.abs(segment[idx]) > 0.02 * np.std(sig):
            return start + idx
        return None

    # ------------------------------------------------------------------
    # Ordering validation
    # ------------------------------------------------------------------

    _ORDERING = [
        "P_onset", "P_peak", "P_offset",
        "QRS_onset", "R_peak", "QRS_offset",
        "T_onset", "T_peak", "T_offset",
    ]

    def _validate_ordering(
        self, fiducials: dict[str, FiducialPoint],
    ) -> dict[str, FiducialPoint]:
        valid: dict[str, FiducialPoint] = {}
        last = -1
        for name in self._ORDERING:
            if name not in fiducials:
                continue
            fp = fiducials[name]
            if fp.sample > last:
                valid[name] = fp
                last = fp.sample
        return valid

    # ------------------------------------------------------------------
    # Intervals
    # ------------------------------------------------------------------

    def _compute_intervals(
        self, fiducials: dict[str, FiducialPoint],
    ) -> dict[str, float]:
        intervals: dict[str, float] = {}

        def _ms(a: str, b: str) -> float | None:
            if a in fiducials and b in fiducials:
                return fiducials[b].time_ms - fiducials[a].time_ms
            return None

        pr = _ms("P_onset", "QRS_onset")
        if pr is not None and pr > 0:
            intervals["pr_interval_ms"] = round(pr, 2)

        qrs = _ms("QRS_onset", "QRS_offset")
        if qrs is not None and qrs > 0:
            intervals["qrs_duration_ms"] = round(qrs, 2)

        qt = _ms("QRS_onset", "T_offset")
        if qt is not None and qt > 0:
            intervals["qt_interval_ms"] = round(qt, 2)

        p_dur = _ms("P_onset", "P_offset")
        if p_dur is not None and p_dur > 0:
            intervals["p_duration_ms"] = round(p_dur, 2)

        t_dur = _ms("T_onset", "T_offset")
        if t_dur is not None and t_dur > 0:
            intervals["t_duration_ms"] = round(t_dur, 2)

        return intervals

    # ------------------------------------------------------------------
    # Morphology
    # ------------------------------------------------------------------

    def _extract_morphology(
        self,
        signal: np.ndarray,
        fiducials: dict[str, FiducialPoint],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        morph: dict[str, Any] = {}
        beat_sig = signal[start:end]
        morph["amplitude_range"] = round(float(beat_sig.max() - beat_sig.min()), 4)

        if "R_peak" in fiducials:
            r_idx = fiducials["R_peak"].sample
            if 0 <= r_idx < len(signal):
                morph["r_amplitude"] = round(float(signal[r_idx]), 4)

        morph["p_wave_present"] = "P_peak" in fiducials

        qrs_on = fiducials.get("QRS_onset")
        qrs_off = fiducials.get("QRS_offset")
        if qrs_on is not None and qrs_off is not None:
            w = qrs_off.time_ms - qrs_on.time_ms
            morph["qrs_wide"] = w > 120.0
        else:
            morph["qrs_wide"] = False

        return morph

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_beat(
        self,
        fiducials: dict[str, FiducialPoint],
        intervals: dict[str, float],
        morphology: dict[str, Any],
    ) -> tuple[str, list[str], float]:
        anomalies: list[str] = []
        has_p = "P_peak" in fiducials
        wide_qrs = morphology.get("qrs_wide", False)

        if wide_qrs and not has_p:
            return "pvc", ["wide_qrs", "absent_p_wave"], 0.7

        pr = intervals.get("pr_interval_ms")
        if has_p and pr is not None and pr < 80.0:
            return "pac", ["short_pr"], 0.5

        if wide_qrs and has_p:
            anomalies.append("wide_qrs")

        if has_p and not wide_qrs:
            return "normal", anomalies, 0.0

        if not has_p:
            anomalies.append("absent_p_wave")
            return "unclassified", anomalies, 0.3

        return "normal", anomalies, 0.0
