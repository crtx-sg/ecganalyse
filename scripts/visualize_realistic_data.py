#!/usr/bin/env python3
"""
Visualize ECG strips from HDF5 files or direct simulator generation.

Usage:
    # From HDF5 file
    python scripts/visualize_realistic_data.py --hdf5 data/samples/PT1234_2024-01.h5
    python scripts/visualize_realistic_data.py --hdf5 data/samples/PT1234_2024-01.h5 --all-leads
    python scripts/visualize_realistic_data.py --hdf5 data/samples/PT1234_2024-01.h5 --event event_1002 --leads ECG1 aVF
    python scripts/visualize_realistic_data.py --hdf5 data/samples/PT1234_2024-01.h5 --duration 5

    # Direct generation (no HDF5 needed)
    python scripts/visualize_realistic_data.py --direct --condition ATRIAL_FIBRILLATION --seed 42

    # Default: auto-detect sample file, or fall back to --direct
    python scripts/visualize_realistic_data.py
"""

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulator.ecg_simulator import ECGSimulator
from src.simulator.conditions import Condition

ALL_LEADS = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]
DEFAULT_LEAD = "ECG2"
DEFAULT_OUTPUT = "ecg_visualization.png"
DEFAULT_SAMPLE = "data/samples/PT1234_2024-01.h5"


def _plot_leads(lead_signals, lead_names, fs, duration_secs, title, output_path):
    """Plot selected ECG leads as vertically stacked subplots sharing the x-axis.

    Args:
        lead_signals: dict mapping lead name -> 1-D numpy array
        lead_names: list of lead names to plot (order = top to bottom)
        fs: sampling rate in Hz
        duration_secs: seconds to display (None = full signal)
        title: figure title
        output_path: where to save the PNG
    """
    num_leads = len(lead_names)
    fig, axes = plt.subplots(
        num_leads, 1,
        figsize=(14, 2.5 * num_leads + 1),
        sharex=True,
        squeeze=False,
    )

    for idx, lead in enumerate(lead_names):
        signal = lead_signals[lead]
        total_samples = len(signal)

        if duration_secs is not None:
            n = min(int(duration_secs * fs), total_samples)
        else:
            n = total_samples

        t = np.arange(n) / fs
        ax = axes[idx, 0]
        ax.plot(t, signal[:n], linewidth=0.8, color="#1a1a2e")
        ax.set_ylabel(lead, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_facecolor("#f8f9fa")

    axes[-1, 0].set_xlabel("Time (seconds)", fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)


def visualize_hdf5(hdf5_path, event_id, leads, all_leads, duration, output):
    """Visualize ECG leads from an HDF5 file."""
    if not os.path.exists(hdf5_path):
        print(f"Error: file not found: {hdf5_path}")
        sys.exit(1)

    with h5py.File(hdf5_path, "r") as f:
        # Resolve event
        event_ids = sorted(k for k in f.keys() if k.startswith("event_"))
        if not event_ids:
            print("Error: no events found in HDF5 file")
            sys.exit(1)

        if event_id is not None:
            if event_id not in f:
                print(f"Error: event '{event_id}' not found. Available: {event_ids}")
                sys.exit(1)
        else:
            event_id = event_ids[0]

        event = f[event_id]

        # Read metadata
        condition = event.attrs.get("condition", "unknown")
        heart_rate = event.attrs.get("heart_rate", 0.0)
        fs = float(f["metadata"]["sampling_rate_ecg"][()])

        # Derive patient ID from filename or metadata
        patient_id = os.path.basename(hdf5_path).split("_")[0]
        if "metadata" in f and "patient_id" in f["metadata"]:
            raw = f["metadata"]["patient_id"][()]
            patient_id = raw.decode() if isinstance(raw, bytes) else str(raw)

        # Determine which leads to plot
        if all_leads:
            lead_names = [l for l in ALL_LEADS if l in event["ecg"]]
        elif leads:
            missing = [l for l in leads if l not in event["ecg"]]
            if missing:
                available = list(event["ecg"].keys())
                print(f"Error: leads not found: {missing}. Available: {available}")
                sys.exit(1)
            lead_names = leads
        else:
            lead_names = [DEFAULT_LEAD]

        # Read signals
        lead_signals = {l: event["ecg"][l][:] for l in lead_names}

        title = (
            f"{condition}  |  HR {heart_rate:.0f} bpm  |  "
            f"{event_id}  |  {patient_id}"
        )

    _plot_leads(lead_signals, lead_names, fs, duration, title, output)


def visualize_direct(condition_name, noise_level, seed, leads, all_leads, duration, output):
    """Generate ECG with ECGSimulator and visualize."""
    condition = Condition[condition_name]
    sim = ECGSimulator(seed=seed)
    event = sim.generate_event(condition=condition, noise_level=noise_level)

    fs = sim.fs

    if all_leads:
        lead_names = ALL_LEADS
    elif leads:
        lead_names = leads
    else:
        lead_names = [DEFAULT_LEAD]

    lead_signals = {l: event.ecg_signals[l] for l in lead_names}

    title = (
        f"{condition.name}  |  HR {event.hr:.0f} bpm  |  "
        f"noise={event.noise_level}  |  seed={seed}"
    )

    _plot_leads(lead_signals, lead_names, fs, duration, title, output)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ECG strips from HDF5 files or direct generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--hdf5", metavar="FILE", help="HDF5 file to visualize")
    mode.add_argument(
        "--direct", action="store_true",
        help="Generate with ECGSimulator (no HDF5 needed)",
    )

    # HDF5 options
    parser.add_argument("--event", metavar="EVENT_ID", help="Event to visualize (default: first)")

    # Lead selection
    parser.add_argument("--leads", nargs="+", metavar="LEAD", help="Leads to plot (default: ECG2)")
    parser.add_argument("--all-leads", action="store_true", help="Show all 7 leads")

    # Display
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Seconds to display (default: full 12s)",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output path (default: {DEFAULT_OUTPUT})")

    # Direct-mode options
    parser.add_argument(
        "--condition", default="NORMAL_SINUS",
        choices=[c.name for c in Condition],
        help="Condition for --direct mode (default: NORMAL_SINUS)",
    )
    parser.add_argument(
        "--noise-level", default="medium",
        choices=["clean", "low", "medium", "high"],
        help="Noise level for --direct mode (default: medium)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --direct mode")

    args = parser.parse_args()

    if args.hdf5:
        visualize_hdf5(
            args.hdf5, args.event, args.leads, args.all_leads,
            args.duration, args.output,
        )
    elif args.direct:
        visualize_direct(
            args.condition, args.noise_level, args.seed,
            args.leads, args.all_leads, args.duration, args.output,
        )
    else:
        # Default: try sample file, fall back to direct
        sample = os.path.join(os.path.dirname(__file__), "..", DEFAULT_SAMPLE)
        if os.path.exists(sample):
            print(f"Auto-detected sample file: {DEFAULT_SAMPLE}")
            visualize_hdf5(
                sample, args.event, args.leads, args.all_leads,
                args.duration, args.output,
            )
        else:
            print("No sample file found, falling back to --direct mode")
            visualize_direct(
                args.condition, args.noise_level, args.seed,
                args.leads, args.all_leads, args.duration, args.output,
            )


if __name__ == "__main__":
    main()
