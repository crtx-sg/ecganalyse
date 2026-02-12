#!/usr/bin/env python3
"""Generate test HDF5 files for multiple cardiac conditions.

Creates one file per condition with clean and medium noise events.

Usage:
    python scripts/generate_conditions_test_set.py
    python scripts/generate_conditions_test_set.py --noise medium
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulator.ecg_simulator import ECGSimulator
from src.simulator.conditions import Condition, CONDITION_REGISTRY
from src.simulator.hdf5_writer import HDF5EventWriter


# Conditions to test with representative HRs
TEST_CONDITIONS = [
    # (condition, target_hr, label)
    (Condition.SINUS_BRADYCARDIA, 45.0, "sinus_brady_45"),
    (Condition.NORMAL_SINUS, 75.0, "normal_sinus_75"),
    (Condition.SINUS_TACHYCARDIA, 130.0, "sinus_tachy_130"),
    (Condition.ATRIAL_FIBRILLATION, 120.0, "afib_120"),
    (Condition.ATRIAL_FLUTTER, 100.0, "aflutter_100"),
    (Condition.SVT, 180.0, "svt_180"),
    (Condition.PVC, 80.0, "pvc_80"),
    (Condition.VENTRICULAR_TACHYCARDIA, 150.0, "vtach_150"),
    (Condition.LBBB, 75.0, "lbbb_75"),
    (Condition.RBBB, 75.0, "rbbb_75"),
    (Condition.AV_BLOCK_1, 70.0, "avblock1_70"),
    (Condition.AV_BLOCK_2_TYPE1, 55.0, "avblock2t1_55"),
    (Condition.ST_ELEVATION, 90.0, "ste_90"),
]


def main():
    parser = argparse.ArgumentParser(description="Generate cardiac condition test files")
    parser.add_argument("--output-dir", default="data/test_conditions",
                        help="Output directory")
    parser.add_argument("--noise", default="medium", choices=["clean", "low", "medium", "high"],
                        help="Noise level (default: medium)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    writer = HDF5EventWriter()

    for condition, hr, label in TEST_CONDITIONS:
        sim = ECGSimulator(seed=args.seed)
        event = sim.generate_event(
            condition=condition,
            hr=hr,
            noise_level=args.noise,
        )
        patient = f"PT_{label[:8].upper()}"
        filepath = os.path.join(args.output_dir, f"{label}_{args.noise}.h5")
        writer.write_file(filepath, [event], patient_id=patient)

        cfg = CONDITION_REGISTRY[condition]
        print(f"  {label:25s}  HR={hr:5.0f}  P-wave={cfg.p_wave_presence:.2f}  "
              f"RR-irreg={cfg.rr_irregularity:.2f}  wide_QRS={cfg.wide_qrs}  "
              f"noise={args.noise}")

    print(f"\n{len(TEST_CONDITIONS)} files written to {args.output_dir}/")


if __name__ == "__main__":
    main()
