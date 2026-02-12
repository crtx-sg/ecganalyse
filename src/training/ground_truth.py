"""Ground truth extraction re-exported from simulator for convenience.

The canonical BeatFiducials dataclass and fiducial-generating function live
in :mod:`src.simulator.morphology` so that the same RNG path produces both
the signal and the ground truth.  This module simply re-exports them for
use by the training pipeline.
"""

from __future__ import annotations

from src.simulator.morphology import BeatFiducials, generate_lead_with_fiducials

__all__ = ["BeatFiducials", "generate_lead_with_fiducials"]
