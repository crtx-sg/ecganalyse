#!/usr/bin/env python3
"""CLI for the ECG simulator — generate MIT-BIH-like HDF5 data files.

Usage examples:
    python scripts/ecg_simulator_cli.py 5 --seed 42
    python scripts/ecg_simulator_cli.py 10 --condition ATRIAL_FIBRILLATION --noise-level high
    python scripts/ecg_simulator_cli.py 20 --balanced --output-dir data/train
    python scripts/ecg_simulator_cli.py 30 --mit-bih
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.simulator.conditions import Condition, CONDITION_REGISTRY
from src.simulator.ecg_simulator import ECGSimulator
from src.simulator.hdf5_writer import HDF5EventWriter


# MIT-BIH-like distribution (approximation of real-world prevalence)
MIT_BIH_PROPORTIONS: dict[Condition, float] = {
    Condition.NORMAL_SINUS: 0.30,
    Condition.SINUS_BRADYCARDIA: 0.05,
    Condition.SINUS_TACHYCARDIA: 0.05,
    Condition.ATRIAL_FIBRILLATION: 0.10,
    Condition.ATRIAL_FLUTTER: 0.03,
    Condition.PAC: 0.05,
    Condition.SVT: 0.03,
    Condition.PVC: 0.10,
    Condition.VENTRICULAR_TACHYCARDIA: 0.05,
    Condition.VENTRICULAR_FIBRILLATION: 0.02,
    Condition.LBBB: 0.05,
    Condition.RBBB: 0.05,
    Condition.AV_BLOCK_1: 0.04,
    Condition.AV_BLOCK_2_TYPE1: 0.03,
    Condition.AV_BLOCK_2_TYPE2: 0.02,
    Condition.ST_ELEVATION: 0.03,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ECG alarm event data in HDF5 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "num_events", type=int, nargs="?", default=5,
        help="Number of alarm events to generate (default 5).",
    )
    parser.add_argument("--patient-id", type=str, help="Patient ID (e.g. PT1234).")
    parser.add_argument(
        "--output-dir", type=str, default="data/samples",
        help="Output directory (default data/samples).",
    )
    parser.add_argument(
        "--condition", type=str, default="random",
        help="Condition name (e.g. ATRIAL_FIBRILLATION) or 'random'.",
    )
    parser.add_argument(
        "--noise-level", type=str, default="medium",
        choices=["clean", "low", "medium", "high"],
        help="Noise level preset (default medium).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    # Convenience presets
    preset = parser.add_argument_group("convenience presets")
    preset.add_argument("--all-normal", action="store_true", help="Only normal sinus.")
    preset.add_argument("--all-abnormal", action="store_true", help="Only abnormal conditions.")
    preset.add_argument("--balanced", action="store_true", help="Uniform across all 16 conditions.")
    preset.add_argument("--mit-bih", action="store_true", help="MIT-BIH-like distribution.")

    return parser


def resolve_proportions(args: argparse.Namespace) -> dict[Condition, float] | None:
    """Determine condition proportions from CLI flags."""
    if args.all_normal:
        return {Condition.NORMAL_SINUS: 1.0}
    if args.all_abnormal:
        abnormal = [c for c in Condition if c != Condition.NORMAL_SINUS]
        return {c: 1.0 / len(abnormal) for c in abnormal}
    if args.balanced:
        return {c: 1.0 / len(Condition) for c in Condition}
    if args.mit_bih:
        return dict(MIT_BIH_PROPORTIONS)
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sim = ECGSimulator(seed=args.seed)
    writer = HDF5EventWriter()

    proportions = resolve_proportions(args)

    # Fixed condition mode
    fixed_condition: Condition | None = None
    if args.condition != "random":
        try:
            fixed_condition = Condition[args.condition]
        except KeyError:
            valid = ", ".join(c.name for c in Condition)
            parser.error(f"Unknown condition '{args.condition}'. Valid: {valid}")

    # Generate events
    events = []
    for i in range(args.num_events):
        cond = fixed_condition
        event = sim.generate_event(
            condition=cond,
            noise_level=args.noise_level,
            condition_proportions=proportions,
        )
        events.append(event)
        print(f"  [{i + 1}/{args.num_events}] {event.condition.name}  HR={event.hr:.1f}")

    # Determine output path
    patient_id = args.patient_id or f"PT{sim._rng.integers(1000, 9999)}"
    date_str = datetime.now().strftime("%Y-%m")
    output_path = Path(args.output_dir) / f"{patient_id}_{date_str}.h5"

    writer.write_file(str(output_path), events, patient_id=patient_id)
    print(f"\nWrote {len(events)} events to {output_path}")


if __name__ == "__main__":
    main()
