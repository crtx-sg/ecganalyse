"""Generate sample HDF5 alarm event files for testing.

Uses the ECG simulator for realistic waveform generation while keeping the
same output paths and deterministic seeding for backward compatibility with
existing test fixtures.

Usage:
    python scripts/generate_sample_hdf5.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.simulator.conditions import Condition
from src.simulator.ecg_simulator import ECGSimulator
from src.simulator.hdf5_writer import HDF5EventWriter


def create_sample_hdf5(
    filepath: str,
    patient_id: str = "PT1234",
    num_events: int = 2,
    condition: Condition = Condition.NORMAL_SINUS,
    hr: float = 72.0,
    seed: int = 42,
) -> None:
    """Create a sample HDF5 alarm event file using the simulator."""
    sim = ECGSimulator(seed=seed)
    writer = HDF5EventWriter()

    events = []
    for _ in range(num_events):
        event = sim.generate_event(
            condition=condition,
            hr=hr,
            noise_level="low",
        )
        events.append(event)

    writer.write_file(filepath, events, patient_id=patient_id)


if __name__ == "__main__":
    output_dir = Path("data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normal sample (same path as before)
    create_sample_hdf5(
        str(output_dir / "PT1234_2024-01.h5"),
        patient_id="PT1234",
        num_events=3,
        condition=Condition.NORMAL_SINUS,
        hr=72.0,
        seed=42,
    )
    print(f"Created {output_dir / 'PT1234_2024-01.h5'}")

    # Tachycardia sample (same path as before)
    create_sample_hdf5(
        str(output_dir / "PT5678_2024-02.h5"),
        patient_id="PT5678",
        num_events=2,
        condition=Condition.SINUS_TACHYCARDIA,
        hr=110.0,
        seed=43,
    )
    print(f"Created {output_dir / 'PT5678_2024-02.h5'}")
