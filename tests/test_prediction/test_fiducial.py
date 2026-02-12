"""Unit tests for FiducialExtractor."""

import numpy as np
import pytest

from src.prediction.fiducial import FiducialExtractor, _FID_IDX, LEAD_ORDER
from src.ecg_system.schemas import Beat, FiducialPoint


FS = 200
N = 2400


def _make_r_peak_heatmap(
    lead_idx: int = 1,
    r_positions: list[int] | None = None,
    peak_value: float = 0.9,
) -> np.ndarray:
    """Create heatmaps with clear R-peaks on a specific lead."""
    if r_positions is None:
        r_positions = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200]

    heatmaps = np.zeros((7, 9, N), dtype=np.float32)
    for t in r_positions:
        for s in range(max(0, t - 10), min(N, t + 10)):
            val = peak_value * np.exp(-((s - t) ** 2) / (2 * 3**2))
            heatmaps[lead_idx, _FID_IDX["R_peak"], s] = val
    return heatmaps


def _make_full_heatmap(lead_idx: int = 1) -> np.ndarray:
    """Create heatmaps with all 9 fiducial types for a regular rhythm."""
    heatmaps = np.zeros((7, 9, N), dtype=np.float32)

    # ~72 bpm, RR ~167 samples
    r_positions = list(range(200, 2200, 167))

    for r_pos in r_positions:
        # Relative positions (samples) from R-peak
        fiducial_offsets = {
            "P_onset": -80,
            "P_peak": -65,
            "P_offset": -50,
            "QRS_onset": -15,
            "R_peak": 0,
            "QRS_offset": 15,
            "T_onset": 40,
            "T_peak": 65,
            "T_offset": 90,
        }
        for fid_name, offset in fiducial_offsets.items():
            pos = r_pos + offset
            if 0 <= pos < N:
                fid_ch = _FID_IDX[fid_name]
                for s in range(max(0, pos - 5), min(N, pos + 5)):
                    val = 0.85 * np.exp(-((s - pos) ** 2) / (2 * 2**2))
                    heatmaps[lead_idx, fid_ch, s] = max(heatmaps[lead_idx, fid_ch, s], val)

    return heatmaps


class TestFiducialExtractor:

    def setup_method(self) -> None:
        self.extractor = FiducialExtractor(fs=FS)

    def test_detects_r_peaks(self) -> None:
        """Should detect R-peaks from heatmap."""
        heatmaps = _make_r_peak_heatmap()
        beats = self.extractor.extract(heatmaps)
        assert len(beats) == 11

    def test_r_peak_in_every_beat(self) -> None:
        """Every beat must have an R_peak fiducial."""
        heatmaps = _make_r_peak_heatmap()
        beats = self.extractor.extract(heatmaps)
        for beat in beats:
            assert "R_peak" in beat.fiducials

    def test_r_peak_confidence(self) -> None:
        """R-peak confidence should be > 0.5 for clear peaks."""
        heatmaps = _make_r_peak_heatmap(peak_value=0.9)
        beats = self.extractor.extract(heatmaps)
        for beat in beats:
            assert beat.fiducials["R_peak"].confidence > 0.5

    def test_fiducial_ordering(self) -> None:
        """Fiducials within a beat must be temporally ordered."""
        heatmaps = _make_full_heatmap()
        beats = self.extractor.extract(heatmaps)
        ordering = [
            "P_onset", "P_peak", "P_offset",
            "QRS_onset", "R_peak", "QRS_offset",
            "T_onset", "T_peak", "T_offset",
        ]
        for beat in beats:
            present = [f for f in ordering if f in beat.fiducials]
            samples = [beat.fiducials[f].sample for f in present]
            assert samples == sorted(samples), f"Beat {beat.beat_index}: ordering violated"

    def test_full_fiducials_detected(self) -> None:
        """Full heatmap should produce beats with most fiducial types."""
        heatmaps = _make_full_heatmap()
        beats = self.extractor.extract(heatmaps)
        # At least some beats should have P + QRS + T
        full_beats = [b for b in beats if len(b.fiducials) >= 5]
        assert len(full_beats) > 0

    def test_beat_type_normal(self) -> None:
        """Beats with P-wave and narrow QRS should be classified as normal."""
        heatmaps = _make_full_heatmap()
        beats = self.extractor.extract(heatmaps)
        normal_beats = [b for b in beats if b.beat_type == "normal"]
        assert len(normal_beats) > 0

    def test_missing_p_wave(self) -> None:
        """Beats without P-wave should not be classified as normal."""
        heatmaps = _make_r_peak_heatmap()  # only R-peaks, no P-wave
        beats = self.extractor.extract(heatmaps)
        for beat in beats:
            assert beat.beat_type != "normal"

    def test_beat_index_sequential(self) -> None:
        """Beat indices should be sequential starting from 0."""
        heatmaps = _make_r_peak_heatmap()
        beats = self.extractor.extract(heatmaps)
        for i, beat in enumerate(beats):
            assert beat.beat_index == i

    def test_lead_assignment(self) -> None:
        """All beats should be assigned to the primary lead."""
        heatmaps = _make_r_peak_heatmap()
        beats = self.extractor.extract(heatmaps)
        for beat in beats:
            assert beat.lead == "ECG2"

    def test_intervals_computed(self) -> None:
        """Beats with full fiducials should have computed intervals."""
        heatmaps = _make_full_heatmap()
        beats = self.extractor.extract(heatmaps)
        beats_with_intervals = [b for b in beats if len(b.intervals) > 0]
        assert len(beats_with_intervals) > 0

    def test_empty_heatmap_no_beats(self) -> None:
        """Zero heatmap should produce no beats."""
        heatmaps = np.zeros((7, 9, N), dtype=np.float32)
        beats = self.extractor.extract(heatmaps)
        assert len(beats) == 0

    def test_time_ms_consistent(self) -> None:
        """time_ms should equal sample / fs * 1000."""
        heatmaps = _make_r_peak_heatmap()
        beats = self.extractor.extract(heatmaps)
        for beat in beats:
            for name, fp in beat.fiducials.items():
                expected_ms = round(fp.sample / FS * 1000.0, 2)
                assert fp.time_ms == expected_ms, f"{name}: {fp.time_ms} != {expected_ms}"
