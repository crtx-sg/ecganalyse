"""Generate Gaussian heatmap ground truth from fiducial positions.

Converts a list of BeatFiducials into a [9, n_samples] tensor where each
channel contains Gaussian peaks at the corresponding fiducial positions.
"""

from __future__ import annotations

import numpy as np
import torch

from src.simulator.morphology import BeatFiducials


# Fiducial channel order (matches heatmap.py)
FIDUCIAL_ATTRS = [
    "p_onset", "p_peak", "p_offset",
    "qrs_onset", "r_peak", "qrs_offset",
    "t_onset", "t_peak", "t_offset",
]
NUM_FIDUCIALS = len(FIDUCIAL_ATTRS)


def fiducials_to_heatmaps(
    beats: list[BeatFiducials],
    n_samples: int = 2400,
    sigma: float = 4.0,
) -> torch.Tensor:
    """Convert beat fiducials to Gaussian heatmap ground truth.

    Args:
        beats: List of BeatFiducials from the simulator.
        n_samples: Signal length (default 2400 for 12s at 200Hz).
        sigma: Gaussian standard deviation in samples.
            4.0 samples = 20ms at 200Hz (FWHM ~47ms).

    Returns:
        Tensor of shape [9, n_samples] with values in [0, 1].
    """
    heatmaps = np.zeros((NUM_FIDUCIALS, n_samples), dtype=np.float32)
    x = np.arange(n_samples, dtype=np.float32)

    for beat in beats:
        for ch_idx, attr in enumerate(FIDUCIAL_ATTRS):
            pos = getattr(beat, attr)
            if pos is None:
                continue
            if pos < 0 or pos >= n_samples:
                continue
            # Additive Gaussian — max-clamp after all beats
            gaussian = np.exp(-0.5 * ((x - pos) / sigma) ** 2)
            heatmaps[ch_idx] += gaussian

    # Clamp to [0, 1] — overlapping beats might exceed 1.0
    np.clip(heatmaps, 0.0, 1.0, out=heatmaps)

    return torch.from_numpy(heatmaps)


def generate_all_lead_heatmaps(
    beats: list[BeatFiducials],
    num_leads: int = 7,
    n_samples: int = 2400,
    sigma: float = 4.0,
) -> torch.Tensor:
    """Generate heatmaps for all leads.

    For synthetic data, fiducial positions are identical across leads
    (all derived from the same beat times), so we replicate.

    Args:
        beats: List of BeatFiducials.
        num_leads: Number of ECG leads (default 7).
        n_samples: Signal length.
        sigma: Gaussian standard deviation.

    Returns:
        Tensor of shape [num_leads, 9, n_samples].
    """
    single = fiducials_to_heatmaps(beats, n_samples, sigma)
    return single.unsqueeze(0).expand(num_leads, -1, -1).clone()
