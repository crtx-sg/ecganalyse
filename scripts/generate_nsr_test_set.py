#!/usr/bin/env python3
"""Generate clean Normal Sinus Rhythm test HDF5 files at exact heart rates.

Creates three HDF5 files with zero noise:
  - NSR_60bpm.h5  (lower boundary of normal)
  - NSR_80bpm.h5  (mid-range normal)
  - NSR_100bpm.h5 (upper boundary of normal)

Each file has a single event (event_1001) with clean ECG, vitals matching
the target HR, and all 7 leads.

Usage:
    python scripts/generate_nsr_test_set.py
    python scripts/generate_nsr_test_set.py --output-dir data/test_nsr
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulator.ecg_simulator import ECGSimulator
from src.simulator.conditions import Condition
from src.simulator.hdf5_writer import HDF5EventWriter


def main():
    parser = argparse.ArgumentParser(description="Generate clean NSR test files")
    parser.add_argument("--output-dir", default="data/test_nsr",
                        help="Output directory (default: data/test_nsr)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    writer = HDF5EventWriter()

    test_cases = [
        {"hr": 60.0,  "label": "60bpm",  "patient": "PT_NSR60"},
        {"hr": 80.0,  "label": "80bpm",  "patient": "PT_NSR80"},
        {"hr": 100.0, "label": "100bpm", "patient": "PT_NSR100"},
    ]

    for tc in test_cases:
        sim = ECGSimulator(seed=args.seed)
        event = sim.generate_event(
            condition=Condition.NORMAL_SINUS,
            hr=tc["hr"],
            noise_level="clean",
        )

        filepath = os.path.join(args.output_dir, f"NSR_{tc['label']}.h5")
        writer.write_file(filepath, [event], patient_id=tc["patient"])
        print(f"Generated: {filepath}  (HR={tc['hr']:.0f} bpm, condition=NORMAL_SINUS, noise=clean)")

    print(f"\nAll files written to {args.output_dir}/")
    print("\nTest commands:")
    for tc in test_cases:
        f = os.path.join(args.output_dir, f"NSR_{tc['label']}.h5")
        print(f"  python scripts/visualize_phase4.py {f} event_1001 --all-leads --output phase4_nsr_{tc['label']}.png")


if __name__ == "__main__":
    main()
