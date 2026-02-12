#!/usr/bin/env python3
"""Phase 3 CLI harness: HDF5 → preprocess → encode → decode → fiducials → JSON.

Usage:
    python scripts/cli_phase3.py <hdf5_file> <event_id>
    python scripts/cli_phase3.py data/samples/PT1234_2024-01.h5 event_1001
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
from src.prediction.heatmap import HeatmapDecoder
from src.prediction.fiducial import FiducialExtractor


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
    ecg_tensor = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
    with torch.no_grad():
        denoised = denoiser(ecg_tensor)

    # Phase 2: Encode
    encoder = FoundationModelAdapter(output_dim=256)
    encoder.eval()
    with torch.no_grad():
        features = encoder(denoised)

    # Phase 3: Decode heatmaps → extract fiducials
    decoder = HeatmapDecoder(d_model=256)
    decoder.eval()
    with torch.no_grad():
        heatmaps = decoder(features)

    hm_np = heatmaps.squeeze(0).numpy()
    ecg_np = event.ecg.as_array

    extractor = FiducialExtractor()
    beats = extractor.extract(hm_np, ecg_np)

    # Build JSON output
    beats_json = []
    for beat in beats:
        fid_dict = {}
        for name, fp in beat.fiducials.items():
            fid_dict[name] = {
                "sample": fp.sample,
                "time_ms": fp.time_ms,
                "confidence": round(fp.confidence, 4),
            }
        beats_json.append({
            "beat_index": beat.beat_index,
            "beat_type": beat.beat_type,
            "lead": beat.lead,
            "fiducials": fid_dict,
            "intervals": beat.intervals,
            "morphology": beat.morphology,
            "anomalies": beat.anomalies,
            "anomaly_confidence": round(beat.anomaly_confidence, 4),
        })

    output = {
        "event_id": event.event_id,
        "quality_sqi": quality_report.overall_sqi,
        "num_beats": len(beats),
        "beats": beats_json,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
