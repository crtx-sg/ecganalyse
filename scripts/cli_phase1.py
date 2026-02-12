#!/usr/bin/env python3
"""Phase 1 CLI harness: load HDF5 event → quality assessment + denoising → JSON report.

Usage:
    python scripts/cli_phase1.py <hdf5_file> <event_id>
    python scripts/cli_phase1.py data/samples/PT1234_2024-01.h5 event_1001
"""

import json
import sys
import os

import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.quality import SignalQualityAssessor
from src.preprocessing.denoiser import ECGDenoiser


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <hdf5_file> <event_id>", file=sys.stderr)
        sys.exit(1)

    hdf5_path = sys.argv[1]
    event_id = sys.argv[2]

    # Load event via Phase 0 loader
    loader = HDF5AlarmEventLoader()
    with loader.load_file(hdf5_path) as f:
        event = loader.load_event(f, event_id)

    # Quality assessment
    assessor = SignalQualityAssessor()
    report = assessor.assess(event.ecg)

    # Denoising
    denoiser = ECGDenoiser()
    denoiser.eval()

    ecg_array = event.ecg.as_array  # [7, 2400]
    x = torch.from_numpy(ecg_array).unsqueeze(0).float()  # [1, 7, 2400]
    with torch.no_grad():
        y = denoiser(x)
    denoised = y.squeeze(0).numpy()  # [7, 2400]

    # Build denoising summary
    lead_order = event.ecg.LEAD_ORDER
    denoising_summary: dict[str, dict[str, float]] = {}
    for i, lead in enumerate(lead_order):
        rms_before = float(np.sqrt(np.mean(ecg_array[i] ** 2)))
        rms_after = float(np.sqrt(np.mean(denoised[i] ** 2)))
        denoising_summary[lead] = {
            "rms_before": round(rms_before, 4),
            "rms_after": round(rms_after, 4),
        }

    # Assemble output
    output = {
        "event_id": event.event_id,
        "quality_report": {
            "overall_sqi": report.overall_sqi,
            "lead_sqi": report.lead_sqi,
            "usable_leads": report.usable_leads,
            "excluded_leads": report.excluded_leads,
            "quality_flags": report.quality_flags,
            "noise_level": report.noise_level,
            "baseline_stability": report.baseline_stability,
        },
        "denoising_summary": denoising_summary,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
