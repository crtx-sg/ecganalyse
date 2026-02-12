#!/usr/bin/env python3
"""Generate noisy Normal Sinus Rhythm test HDF5 files.

Creates 9 files: 3 heart rates x 3 noise levels.

Usage:
    python scripts/generate_nsr_noisy_test_set.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulator.ecg_simulator import ECGSimulator
from src.simulator.conditions import Condition
from src.simulator.hdf5_writer import HDF5EventWriter


def main():
    parser = argparse.ArgumentParser(description="Generate noisy NSR test files")
    parser.add_argument("--output-dir", default="data/test_nsr_noisy",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    writer = HDF5EventWriter()

    heart_rates = [60.0, 80.0, 100.0]
    noise_levels = ["low", "medium", "high"]

    for hr in heart_rates:
        for noise in noise_levels:
            sim = ECGSimulator(seed=args.seed)
            event = sim.generate_event(
                condition=Condition.NORMAL_SINUS,
                hr=hr,
                noise_level=noise,
            )
            label = f"NSR_{int(hr)}bpm_{noise}"
            patient = f"PT_N{int(hr)}{noise[0].upper()}"
            filepath = os.path.join(args.output_dir, f"{label}.h5")
            writer.write_file(filepath, [event], patient_id=patient)
            print(f"Generated: {filepath}  (HR={hr:.0f}, noise={noise})")

    print(f"\nAll {len(heart_rates) * len(noise_levels)} files in {args.output_dir}/")


if __name__ == "__main__":
    main()
