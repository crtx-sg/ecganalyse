#!/usr/bin/env python3
"""Phase 2 CLI harness: load HDF5 → preprocess → encode → JSON feature stats.

Uses ECG-TransCovNet encoder (requires trained weights).

Usage:
    python scripts/cli_phase2.py <hdf5_file> <event_id>
    python scripts/cli_phase2.py data/samples/PT1234_2024-01.h5 event_1001
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.quality import SignalQualityAssessor
from src.preprocessing.denoiser import ECGDenoiser
from src.encoding.foundation import FoundationModelAdapter

LEAD_ORDER = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]

_TRANSCOVNET_WEIGHTS = (
    Path(__file__).resolve().parents[1] / "models" / "ecg_transcovnet" / "best_model.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: ECG feature encoding")
    parser.add_argument("hdf5_file", help="Path to HDF5 alarm event file")
    parser.add_argument("event_id", help="Event ID (e.g. event_1001)")
    args = parser.parse_args()

    if not _TRANSCOVNET_WEIGHTS.exists():
        print(f"Error: TransCovNet weights not found at {_TRANSCOVNET_WEIGHTS}", file=sys.stderr)
        print("Run train_ecg_transcovnet.py first.", file=sys.stderr)
        sys.exit(1)

    # Phase 0: Load
    loader = HDF5AlarmEventLoader()
    with loader.load_file(args.hdf5_file) as f:
        event = loader.load_event(f, args.event_id)

    # Phase 1: Quality + Denoise
    assessor = SignalQualityAssessor()
    quality_report = assessor.assess(event.ecg)

    denoiser = ECGDenoiser()
    denoiser.eval()
    ecg_array = event.ecg.as_array  # [7, 2400]
    x = torch.from_numpy(ecg_array).unsqueeze(0).float()
    with torch.no_grad():
        denoised = denoiser(x)  # [1, 7, 2400]

    # Phase 2: Encode (TransCovNet only)
    encoder = FoundationModelAdapter(output_dim=256, model_type="transcovnet")
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
        "encoder_type": encoder.model_type,
        "encoder_output": {
            "shape": list(features_np.shape),
            "dtype": str(features_np.dtype),
            "per_lead_stats": lead_stats,
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
