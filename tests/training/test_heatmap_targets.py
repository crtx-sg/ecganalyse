"""Tests for heatmap ground truth generation."""

import torch
import pytest

from src.simulator.morphology import BeatFiducials
from src.training.heatmap_targets import (
    fiducials_to_heatmaps,
    generate_all_lead_heatmaps,
    NUM_FIDUCIALS,
    FIDUCIAL_ATTRS,
)


@pytest.fixture
def sample_fiducials():
    """Create sample fiducials for 2 beats."""
    return [
        BeatFiducials(
            p_onset=140, p_peak=150, p_offset=160,
            qrs_onset=180, r_peak=200, qrs_offset=220,
            t_onset=260, t_peak=280, t_offset=300,
        ),
        BeatFiducials(
            p_onset=540, p_peak=550, p_offset=560,
            qrs_onset=580, r_peak=600, qrs_offset=620,
            t_onset=660, t_peak=680, t_offset=700,
        ),
    ]


class TestFiducialsToHeatmaps:
    def test_output_shape(self, sample_fiducials):
        hm = fiducials_to_heatmaps(sample_fiducials)
        assert hm.shape == (9, 2400)

    def test_output_dtype(self, sample_fiducials):
        hm = fiducials_to_heatmaps(sample_fiducials)
        assert hm.dtype == torch.float32

    def test_values_in_range(self, sample_fiducials):
        hm = fiducials_to_heatmaps(sample_fiducials)
        assert hm.min() >= 0.0
        assert hm.max() <= 1.0

    def test_peaks_at_fiducial_positions(self, sample_fiducials):
        """Heatmap peaks should be at fiducial positions."""
        hm = fiducials_to_heatmaps(sample_fiducials, sigma=4.0)

        # R-peak channel (idx 4) should peak near sample 200
        r_channel = hm[4]
        peak_pos = r_channel.argmax().item()
        assert abs(peak_pos - 200) <= 1

    def test_absent_fiducials_are_zero(self):
        """Channels with no fiducials should be all zeros."""
        beats = [
            BeatFiducials(
                p_onset=None, p_peak=None, p_offset=None,
                qrs_onset=180, r_peak=200, qrs_offset=220,
                t_onset=260, t_peak=280, t_offset=300,
            ),
        ]
        hm = fiducials_to_heatmaps(beats)

        # P-wave channels (0, 1, 2) should be zero
        assert hm[0].sum().item() == 0.0
        assert hm[1].sum().item() == 0.0
        assert hm[2].sum().item() == 0.0

        # QRS channels should not be zero
        assert hm[4].sum().item() > 0.0

    def test_sigma_controls_width(self, sample_fiducials):
        """Larger sigma should produce wider peaks."""
        hm_narrow = fiducials_to_heatmaps(sample_fiducials, sigma=2.0)
        hm_wide = fiducials_to_heatmaps(sample_fiducials, sigma=8.0)

        # Count samples above 0.5 threshold in R-peak channel
        narrow_width = (hm_narrow[4] > 0.5).sum().item()
        wide_width = (hm_wide[4] > 0.5).sum().item()
        assert wide_width > narrow_width

    def test_empty_fiducials(self):
        hm = fiducials_to_heatmaps([])
        assert hm.shape == (9, 2400)
        assert hm.sum().item() == 0.0

    def test_custom_n_samples(self, sample_fiducials):
        hm = fiducials_to_heatmaps(sample_fiducials, n_samples=1000)
        assert hm.shape == (9, 1000)


class TestGenerateAllLeadHeatmaps:
    def test_output_shape(self, sample_fiducials):
        hm = generate_all_lead_heatmaps(sample_fiducials)
        assert hm.shape == (7, 9, 2400)

    def test_all_leads_identical(self, sample_fiducials):
        """For synthetic data, all leads should have identical heatmaps."""
        hm = generate_all_lead_heatmaps(sample_fiducials)
        for i in range(1, 7):
            assert torch.allclose(hm[0], hm[i])

    def test_custom_num_leads(self, sample_fiducials):
        hm = generate_all_lead_heatmaps(sample_fiducials, num_leads=3)
        assert hm.shape == (3, 9, 2400)
