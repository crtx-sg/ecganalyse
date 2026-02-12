#!/usr/bin/env python3
"""Phase 2 CLI harness: load HDF5 → preprocess → encode → JSON feature stats.

Usage:
    python scripts/cli_phase2.py <hdf5_file> <event_id>
    python scripts/cli_phase2.py data/samples/PT1234_2024-01.h5 event_1001
"""

import json
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.quality import SignalQualityAssessor
from src.preprocessing.denoiser import ECGDenoiser
from src.encoding.foundation import FoundationModelAdapter

LEAD_ORDER = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <hdf5_file> <event_id>", file=sys.stderr)
        sys.exit(1)

    hdf5_path = sys.argv[1]
    event_id = sys.argv[2]

    # Phase 0: Load
    loader = HDF5AlarmEventLoader()
    with loader.load_file(hdf5_path) as f:
        event = loader.load_event(f, event_id)

    # Phase 1: Quality + Denoise
    assessor = SignalQualityAssessor()
    quality_report = assessor.assess(event.ecg)

    denoiser = ECGDenoiser()
    denoiser.eval()
    ecg_array = event.ecg.as_array  # [7, 2400]
    x = torch.from_numpy(ecg_array).unsqueeze(0).float()
    with torch.no_grad():
        denoised = denoiser(x)  # [1, 7, 2400]

    # Phase 2: Encode
    encoder = FoundationModelAdapter(output_dim=256)
    encoder.eval()
    with torch.no_grad():
        features = encoder(denoised)  # [1, 7, S, D]

    features_np = features.squeeze(0).numpy()  # [7, S, D]

    # Build per-lead statistics
    lead_stats: dict[str, dict] = {}
    for i, lead in enumerate(LEAD_ORDER):
        lead_feat = features_np[i]  # [S, D]
        lead_stats[lead] = {
            "min": round(float(np.min(lead_feat)), 4),
            "max": round(float(np.max(lead_feat)), 4),
            "mean": round(float(np.mean(lead_feat)), 4),
            "std": round(float(np.std(lead_feat)), 4),
        }

    output = {
        "event_id": event.event_id,
        "quality_sqi": quality_report.overall_sqi,
        "encoder_output": {
            "shape": list(features_np.shape),
            "dtype": str(features_np.dtype),
            "per_lead_stats": lead_stats,
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
