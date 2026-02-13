#!/usr/bin/env python3
"""Phase 4 CLI harness: HDF5 → full pipeline → JSON Feature Assembly.

Defaults to signal-based beat detection (classical DSP) and TransCovNet
condition classification. Use --heatmap to opt into the neural heatmap
decoder for beat detection (requires trained heatmap model weights).

Usage:
    python scripts/cli_phase4.py <hdf5_file> <event_id>
    python scripts/cli_phase4.py data/samples/PT1234_2024-01.h5 event_1001
    python scripts/cli_phase4.py data/samples/PT1234_2024-01.h5 event_1001 --heatmap
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.quality import SignalQualityAssessor
from src.prediction.signal_beat_detector import SignalBasedBeatDetector
from src.prediction.condition_classifier import ConditionClassifier
from src.interpretation.symbolic import SymbolicCalculationEngine
from src.interpretation.rules import RuleBasedReasoningEngine
from src.interpretation.vitals_context import VitalsContextIntegrator
from src.interpretation.assembly import JSONAssembler

_TRANSCOVNET_WEIGHTS = (
    Path(__file__).resolve().parents[1] / "models" / "ecg_transcovnet" / "best_model.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 CLI: full interpretation pipeline")
    parser.add_argument("hdf5_file", help="Path to HDF5 file")
    parser.add_argument("event_id", help="Event ID (e.g. event_1001)")
    parser.add_argument("--heatmap", action="store_true",
                        help="Use neural heatmap decoder for beat detection "
                             "(requires trained heatmap model weights)")
    args = parser.parse_args()

    if not _TRANSCOVNET_WEIGHTS.exists():
        print(f"Error: TransCovNet weights not found at {_TRANSCOVNET_WEIGHTS}", file=sys.stderr)
        print("Run train_ecg_transcovnet.py first.", file=sys.stderr)
        sys.exit(1)

    start_time = time.monotonic()

    # Phase 0: Load
    loader = HDF5AlarmEventLoader()
    with loader.load_file(args.hdf5_file) as f:
        event = loader.load_event(f, args.event_id)

    # Phase 1: Quality
    assessor = SignalQualityAssessor()
    quality = assessor.assess(event.ecg)

    ecg_np = event.ecg.as_array
    ecg_tensor = torch.from_numpy(ecg_np).unsqueeze(0).float()

    # Phase 3a: Beat detection
    if args.heatmap:
        # Neural heatmap pipeline (requires trained decoder weights)
        from src.preprocessing.denoiser import ECGDenoiser
        from src.encoding.foundation import FoundationModelAdapter
        from src.prediction.heatmap import HeatmapDecoder
        from src.prediction.fiducial import FiducialExtractor

        denoiser = ECGDenoiser()
        denoiser.eval()
        with torch.no_grad():
            denoised = denoiser(ecg_tensor)

        encoder = FoundationModelAdapter(output_dim=256, model_type="transcovnet")
        encoder.eval()
        with torch.no_grad():
            features = encoder(denoised)

        decoder = HeatmapDecoder(d_model=256)
        decoder.eval()
        with torch.no_grad():
            heatmaps = decoder(features)

        hm_np = heatmaps.squeeze(0).numpy()
        extractor = FiducialExtractor()
        beats = extractor.extract(hm_np, ecg_np)
    else:
        # Signal-based beat detection (default — no neural model needed)
        detector = SignalBasedBeatDetector(fs=200)
        beats = detector.detect(ecg_np)

    # Phase 3b: Neural condition classification
    classifier = ConditionClassifier()
    condition_pred = classifier.classify(ecg_tensor)
    condition_dict = {
        "condition": condition_pred.condition,
        "rhythm_label": condition_pred.rhythm_label,
        "confidence": condition_pred.confidence,
    }

    # Phase 4: Interpretation (with neural condition augmentation)
    sym = SymbolicCalculationEngine(fs=200)
    measurements = sym.compute_global_measurements(beats)
    rhythm_metrics = sym.compute_rhythm_metrics(beats)

    rules = RuleBasedReasoningEngine()
    rhythm = rules.classify_rhythm(
        measurements, rhythm_metrics, beats,
        condition_prediction=condition_dict,
    )
    findings = rules.generate_findings(measurements, rhythm, beats)

    vitals_integrator = VitalsContextIntegrator()
    vital_findings = vitals_integrator.integrate(measurements, event.vitals)
    findings.extend(vital_findings)

    # Assemble JSON
    assembler = JSONAssembler()
    result = assembler.assemble(
        event, quality, measurements, rhythm, beats,
        findings, sym.traces,
        processing_start_time=start_time,
    )

    # Inject condition classification into result
    result["condition_classification"] = {
        "condition": condition_pred.condition,
        "rhythm_label": condition_pred.rhythm_label,
        "confidence": condition_pred.confidence,
        "top_predictions": [
            {"condition": name, "confidence": conf}
            for name, conf in condition_pred.top_k
        ],
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
