"""Fiducial point extraction from heatmap outputs.

Converts heatmap probabilities into discrete fiducial point locations,
segments beats using R-peak positions, validates physiological plausibility,
and classifies each beat.

Input : heatmaps [batch, 7, 9, 2400] with values in [0, 1]
Output: list[Beat] per event
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks

from src.ecg_system.schemas import Beat, FiducialPoint
from src.prediction.heatmap import FIDUCIAL_NAMES

LEAD_ORDER = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]
FS = 200  # Hz
NUM_SAMPLES = 2400

# Indices into the 9-channel heatmap
_FID_IDX = {name: i for i, name in enumerate(FIDUCIAL_NAMES)}

# Physiological ordering constraints (sample index must increase)
_ORDERING = [
    "P_onset", "P_peak", "P_offset",
    "QRS_onset", "R_peak", "QRS_offset",
    "T_onset", "T_peak", "T_offset",
]


class FiducialExtractor:
    """Extract fiducial points from heatmaps and segment into beats.

    Args:
        fs:                Sampling frequency in Hz (default 200).
        peak_threshold:    Minimum heatmap value to consider a peak (default 0.3).
        min_rr_samples:    Minimum R-R interval in samples (default 80 = 0.4 s).
        primary_lead:      Lead used for beat segmentation (default ``"ECG2"``).
    """

    def __init__(
        self,
        fs: int = FS,
        peak_threshold: float = 0.3,
        min_rr_samples: int = 80,
        primary_lead: str = "ECG2",
    ) -> None:
        self.fs = fs
        self.peak_threshold = peak_threshold
        self.min_rr_samples = min_rr_samples
        self.primary_lead = primary_lead

    def extract(
        self,
        heatmaps: np.ndarray,
        ecg_signals: np.ndarray | None = None,
    ) -> list[Beat]:
        """Extract beats with fiducials from heatmaps.

        Args:
            heatmaps: ``[7, 9, 2400]`` numpy array (single event, no batch dim).
            ecg_signals: Optional ``[7, 2400]`` raw ECG for morphology features.

        Returns:
            List of :class:`Beat` objects ordered by beat index.
        """
        primary_idx = LEAD_ORDER.index(self.primary_lead)
        r_peak_heatmap = heatmaps[primary_idx, _FID_IDX["R_peak"]]

        # Detect R-peaks on the primary lead
        r_peaks = self._detect_peaks(r_peak_heatmap)
        if len(r_peaks) == 0:
            return []

        # Segment beats around R-peaks
        beat_boundaries = self._segment_beats(r_peaks, NUM_SAMPLES)

        beats: list[Beat] = []
        for beat_idx, (start, end, r_sample) in enumerate(beat_boundaries):
            # Extract fiducials for primary lead within this beat window
            fiducials = self._extract_beat_fiducials(
                heatmaps[primary_idx], start, end,
            )

            # Ensure R-peak is present (use the detected one)
            if "R_peak" not in fiducials:
                fiducials["R_peak"] = FiducialPoint(
                    sample=int(r_sample),
                    time_ms=round(r_sample / self.fs * 1000.0, 2),
                    confidence=float(r_peak_heatmap[r_sample]),
                )

            # Validate ordering
            fiducials = self._validate_ordering(fiducials)

            # Compute intervals
            intervals = self._compute_intervals(fiducials)

            # Morphology features
            morphology = self._extract_morphology(
                ecg_signals[primary_idx] if ecg_signals is not None else None,
                fiducials, start, end,
            )

            # Classify beat
            beat_type, anomalies, anomaly_conf = self._classify_beat(
                fiducials, intervals, morphology,
            )

            beats.append(Beat(
                beat_index=beat_idx,
                beat_type=beat_type,
                lead=self.primary_lead,
                fiducials=fiducials,
                intervals=intervals,
                morphology=morphology,
                anomalies=anomalies,
                anomaly_confidence=anomaly_conf,
            ))

        return beats

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------

    def _detect_peaks(self, heatmap: np.ndarray) -> np.ndarray:
        """Detect peaks in a 1-D heatmap."""
        peaks, props = find_peaks(
            heatmap,
            height=self.peak_threshold,
            distance=self.min_rr_samples,
        )
        return peaks

    # ------------------------------------------------------------------
    # Beat segmentation
    # ------------------------------------------------------------------

    def _segment_beats(
        self, r_peaks: np.ndarray, total_samples: int,
    ) -> list[tuple[int, int, int]]:
        """Segment into beat windows centred on R-peaks.

        Returns list of (start, end, r_peak_sample) tuples.
        """
        boundaries: list[tuple[int, int, int]] = []
        for i, r in enumerate(r_peaks):
            if i == 0:
                start = 0
            else:
                start = (r_peaks[i - 1] + r) // 2
            if i == len(r_peaks) - 1:
                end = total_samples
            else:
                end = (r + r_peaks[i + 1]) // 2
            boundaries.append((int(start), int(end), int(r)))
        return boundaries

    # ------------------------------------------------------------------
    # Fiducial extraction per beat
    # ------------------------------------------------------------------

    def _extract_beat_fiducials(
        self,
        lead_heatmaps: np.ndarray,   # [9, 2400]
        start: int,
        end: int,
    ) -> dict[str, FiducialPoint]:
        """Extract fiducial points from heatmaps within a beat window."""
        fiducials: dict[str, FiducialPoint] = {}
        for fid_name, fid_idx in _FID_IDX.items():
            hm = lead_heatmaps[fid_idx, start:end]
            if len(hm) == 0:
                continue
            peaks, props = find_peaks(hm, height=self.peak_threshold)
            if len(peaks) == 0:
                # Use argmax if above threshold
                max_val = float(hm.max())
                if max_val >= self.peak_threshold:
                    peak_local = int(np.argmax(hm))
                    sample = start + peak_local
                    fiducials[fid_name] = FiducialPoint(
                        sample=sample,
                        time_ms=round(sample / self.fs * 1000.0, 2),
                        confidence=max_val,
                    )
            else:
                # Take the highest peak
                best = peaks[np.argmax(props["peak_heights"])]
                sample = start + int(best)
                conf = float(hm[best])
                fiducials[fid_name] = FiducialPoint(
                    sample=sample,
                    time_ms=round(sample / self.fs * 1000.0, 2),
                    confidence=conf,
                )
        return fiducials

    # ------------------------------------------------------------------
    # Ordering validation
    # ------------------------------------------------------------------

    def _validate_ordering(
        self, fiducials: dict[str, FiducialPoint],
    ) -> dict[str, FiducialPoint]:
        """Remove fiducials that violate temporal ordering constraints.

        Expected order: P_onset < P_peak < P_offset < QRS_onset < R_peak
                        < QRS_offset < T_onset < T_peak < T_offset
        """
        ordered = {k: fiducials[k] for k in _ORDERING if k in fiducials}
        valid: dict[str, FiducialPoint] = {}
        last_sample = -1
        for name in _ORDERING:
            if name not in ordered:
                continue
            fp = ordered[name]
            if fp.sample > last_sample:
                valid[name] = fp
                last_sample = fp.sample
            # else: skip — violates ordering
        return valid

    # ------------------------------------------------------------------
    # Interval computation
    # ------------------------------------------------------------------

    def _compute_intervals(
        self, fiducials: dict[str, FiducialPoint],
    ) -> dict[str, float]:
        """Compute clinical intervals in milliseconds."""
        intervals: dict[str, float] = {}

        def _ms(a: str, b: str) -> float | None:
            if a in fiducials and b in fiducials:
                return fiducials[b].time_ms - fiducials[a].time_ms
            return None

        pr = _ms("P_onset", "QRS_onset")
        if pr is not None:
            intervals["pr_interval_ms"] = round(pr, 2)

        qrs = _ms("QRS_onset", "QRS_offset")
        if qrs is not None:
            intervals["qrs_duration_ms"] = round(qrs, 2)

        qt = _ms("QRS_onset", "T_offset")
        if qt is not None:
            intervals["qt_interval_ms"] = round(qt, 2)

        p_dur = _ms("P_onset", "P_offset")
        if p_dur is not None:
            intervals["p_duration_ms"] = round(p_dur, 2)

        t_dur = _ms("T_onset", "T_offset")
        if t_dur is not None:
            intervals["t_duration_ms"] = round(t_dur, 2)

        return intervals

    # ------------------------------------------------------------------
    # Morphology extraction
    # ------------------------------------------------------------------

    def _extract_morphology(
        self,
        signal: np.ndarray | None,
        fiducials: dict[str, FiducialPoint],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        """Extract morphological features from raw signal."""
        morph: dict[str, Any] = {}

        if signal is None:
            return morph

        beat_signal = signal[start:end]
        morph["amplitude_range"] = round(float(beat_signal.max() - beat_signal.min()), 4)

        if "R_peak" in fiducials:
            r_idx = fiducials["R_peak"].sample
            if 0 <= r_idx < len(signal):
                morph["r_amplitude"] = round(float(signal[r_idx]), 4)

        # P-wave present?
        morph["p_wave_present"] = "P_peak" in fiducials

        # QRS width category
        qrs_ms = fiducials.get("QRS_offset", None)
        qrs_onset = fiducials.get("QRS_onset", None)
        if qrs_ms is not None and qrs_onset is not None:
            w = qrs_ms.time_ms - qrs_onset.time_ms
            morph["qrs_wide"] = w > 120.0
        else:
            morph["qrs_wide"] = False

        return morph

    # ------------------------------------------------------------------
    # Beat classification
    # ------------------------------------------------------------------

    def _classify_beat(
        self,
        fiducials: dict[str, FiducialPoint],
        intervals: dict[str, float],
        morphology: dict[str, Any],
    ) -> tuple[str, list[str], float]:
        """Classify beat type and detect anomalies.

        Returns:
            (beat_type, anomalies, anomaly_confidence)
        """
        anomalies: list[str] = []
        conf = 0.0

        has_p = "P_peak" in fiducials
        wide_qrs = morphology.get("qrs_wide", False)

        # PVC: wide QRS + no P-wave
        if wide_qrs and not has_p:
            return "pvc", ["wide_qrs", "absent_p_wave"], 0.7

        # PAC: P-wave present but premature (short PR or abnormal P)
        # Heuristic: if PR interval is very short
        pr = intervals.get("pr_interval_ms")
        if has_p and pr is not None and pr < 80.0:
            return "pac", ["short_pr"], 0.5

        # Wide QRS with P-wave could be bundle branch block — classify as normal
        # for now (Phase 4 interpretation will refine)
        if wide_qrs and has_p:
            anomalies.append("wide_qrs")
            conf = 0.4

        # Normal: has P-wave, narrow QRS
        if has_p and not wide_qrs:
            return "normal", anomalies, conf

        # No P-wave but narrow QRS → unclassified
        if not has_p:
            anomalies.append("absent_p_wave")
            return "unclassified", anomalies, 0.3

        return "normal", anomalies, conf
