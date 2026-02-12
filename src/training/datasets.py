"""On-the-fly synthetic ECG training datasets.

Generates training data by running the simulator for each item, producing
infinite diversity with no disk storage. Each worker gets a deterministic
per-item RNG seed for reproducibility.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.simulator.conditions import Condition, CONDITION_REGISTRY
from src.simulator.ecg_simulator import ECGSimulator, LEAD_NAMES
from src.simulator.noise import NOISE_PRESETS
from src.training.heatmap_targets import generate_all_lead_heatmaps

# Conditions excluded from heatmap training (chaotic, no fiducials)
_EXCLUDED_FROM_HEATMAP = {Condition.VENTRICULAR_FIBRILLATION}

# All trainable conditions (with valid fiducials)
TRAINABLE_CONDITIONS = [c for c in Condition if c not in _EXCLUDED_FROM_HEATMAP]
ALL_CONDITIONS = list(Condition)


class SyntheticECGTrainingDataset(Dataset):
    """On-the-fly synthetic ECG dataset for training.

    Each ``__getitem__`` call runs the simulator with a per-item seed,
    producing a unique but reproducible sample.

    Args:
        epoch_size: Number of samples per "epoch" (default 5000).
        base_seed: Base seed; item seed = base_seed + idx.
        noise_levels: List of noise levels to sample from.
        include_vfib: If False, exclude VFib (no fiducials).
        sigma: Gaussian sigma for heatmap targets.
        fs: Sampling frequency.
        duration: Signal duration in seconds.
    """

    def __init__(
        self,
        epoch_size: int = 5000,
        base_seed: int = 42,
        noise_levels: Optional[list[str]] = None,
        include_vfib: bool = False,
        sigma: float = 4.0,
        fs: float = 200.0,
        duration: float = 12.0,
    ) -> None:
        self.epoch_size = epoch_size
        self.base_seed = base_seed
        self.noise_levels = noise_levels or ["clean", "low", "medium"]
        self.sigma = sigma
        self.fs = fs
        self.duration = duration
        self.n_samples = int(fs * duration)
        self.conditions = ALL_CONDITIONS if include_vfib else TRAINABLE_CONDITIONS

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        seed = self.base_seed + idx
        rng = np.random.default_rng(seed)

        # Random condition and HR
        cond = self.conditions[rng.integers(len(self.conditions))]
        cfg = CONDITION_REGISTRY[cond]
        hr = float(rng.uniform(*cfg.hr_range))

        # Random noise level
        noise_level = self.noise_levels[rng.integers(len(self.noise_levels))]

        # Generate training event
        sim = ECGSimulator(fs=self.fs, duration=self.duration, seed=seed)
        event = sim.generate_training_event(
            condition=cond,
            hr=hr,
            noise_level=noise_level,
        )

        # Stack into tensors [7, n_samples]
        lead_order = LEAD_NAMES
        ecg_noisy = torch.stack([
            torch.from_numpy(event.ecg_noisy[lead]) for lead in lead_order
        ])
        ecg_clean = torch.stack([
            torch.from_numpy(event.ecg_clean[lead]) for lead in lead_order
        ])

        # Generate heatmap ground truth [7, 9, n_samples]
        heatmaps = generate_all_lead_heatmaps(
            event.fiducial_positions,
            num_leads=len(lead_order),
            n_samples=self.n_samples,
            sigma=self.sigma,
        )

        return {
            "ecg_noisy": ecg_noisy,       # [7, 2400]
            "ecg_clean": ecg_clean,        # [7, 2400]
            "heatmaps": heatmaps,          # [7, 9, 2400]
            "condition": cond.value,       # str
            "hr": torch.tensor(hr, dtype=torch.float32),
            "noise_level": noise_level,    # str
        }


def create_validation_dataset(
    size: int = 500,
    seed: int = 99999,
    **kwargs,
) -> SyntheticECGTrainingDataset:
    """Create a fixed validation dataset with a specific seed."""
    return SyntheticECGTrainingDataset(
        epoch_size=size,
        base_seed=seed,
        **kwargs,
    )
