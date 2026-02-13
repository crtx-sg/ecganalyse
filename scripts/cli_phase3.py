#!/usr/bin/env python3
"""Phase 3 CLI harness: HDF5 → preprocess → encode → decode → fiducials → JSON.

Uses ECG-TransCovNet encoder and neural condition classification (requires
trained weights).

Usage:
    python scripts/cli_phase3.py <hdf5_file> <event_id>
    python scripts/cli_phase3.py data/samples/PT1234_2024-01.h5 event_1001
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
from src.prediction.heatmap import HeatmapDecoder
from src.prediction.fiducial import FiducialExtractor
from src.prediction.condition_classifier import ConditionClassifier

_TRANSCOVNET_WEIGHTS = (
    Path(__file__).resolve().parents[1] / "models" / "ecg_transcovnet" / "best_model.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Dense prediction & fiducials")
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
    ecg_tensor = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
    with torch.no_grad():
        denoised = denoiser(ecg_tensor)

    # Phase 2: Encode (TransCovNet only)
    encoder = FoundationModelAdapter(output_dim=256, model_type="transcovnet")
    encoder.eval()
    with torch.no_grad():
        features = encoder(denoised)

    # Phase 3a: Decode heatmaps → extract fiducials
    decoder = HeatmapDecoder(d_model=256)
    decoder.eval()
    with torch.no_grad():
        heatmaps = decoder(features)

    hm_np = heatmaps.squeeze(0).numpy()
    ecg_np = event.ecg.as_array

    extractor = FiducialExtractor()
    beats = extractor.extract(hm_np, ecg_np)

    # Phase 3b: Neural condition classification (uses raw ECG, not denoised)
    classifier = ConditionClassifier()
    pred = classifier.classify(ecg_tensor)
    condition_result = {
        "condition": pred.condition,
        "rhythm_label": pred.rhythm_label,
        "confidence": pred.confidence,
        "top_predictions": [
            {"condition": name, "confidence": conf}
            for name, conf in pred.top_k
        ],
    }

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

    output: dict = {
        "event_id": event.event_id,
        "quality_sqi": quality_report.overall_sqi,
        "encoder_type": encoder.model_type,
        "num_beats": len(beats),
        "beats": beats_json,
        "condition_classification": condition_result,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
