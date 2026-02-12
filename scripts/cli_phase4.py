#!/usr/bin/env python3
"""Phase 4 CLI harness: HDF5 → full pipeline → JSON Feature Assembly.

Usage:
    python scripts/cli_phase4.py <hdf5_file> <event_id>
    python scripts/cli_phase4.py <hdf5_file> <event_id> --signal-based
    python scripts/cli_phase4.py data/samples/PT1234_2024-01.h5 event_1001 --signal-based
"""

import argparse
import json
import sys
import os
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.quality import SignalQualityAssessor
from src.preprocessing.denoiser import ECGDenoiser
from src.encoding.foundation import FoundationModelAdapter
from src.prediction.heatmap import HeatmapDecoder
from src.prediction.fiducial import FiducialExtractor
from src.prediction.signal_beat_detector import SignalBasedBeatDetector
from src.interpretation.symbolic import SymbolicCalculationEngine
from src.interpretation.rules import RuleBasedReasoningEngine
from src.interpretation.vitals_context import VitalsContextIntegrator
from src.interpretation.assembly import JSONAssembler


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 CLI: full interpretation pipeline")
    parser.add_argument("hdf5_file", help="Path to HDF5 file")
    parser.add_argument("event_id", help="Event ID (e.g. event_1001)")
    parser.add_argument("--signal-based", action="store_true",
                        help="Use signal-based beat detection (bypasses neural Phase 2-3)")
    args = parser.parse_args()

    start_time = time.monotonic()

    # Phase 0: Load
    loader = HDF5AlarmEventLoader()
    with loader.load_file(args.hdf5_file) as f:
        event = loader.load_event(f, args.event_id)

    # Phase 1: Quality
    assessor = SignalQualityAssessor()
    quality = assessor.assess(event.ecg)

    ecg_np = event.ecg.as_array

    if args.signal_based:
        # Direct signal-based beat detection
        detector = SignalBasedBeatDetector(fs=200)
        beats = detector.detect(ecg_np)
    else:
        # Neural pipeline: Phase 1 denoise → Phase 2 encode → Phase 3 decode
        denoiser = ECGDenoiser()
        denoiser.eval()
        ecg_tensor = torch.from_numpy(ecg_np).unsqueeze(0).float()
        with torch.no_grad():
            denoised = denoiser(ecg_tensor)

        encoder = FoundationModelAdapter(output_dim=256)
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

    # Phase 4: Interpretation
    sym = SymbolicCalculationEngine(fs=200)
    measurements = sym.compute_global_measurements(beats)
    rhythm_metrics = sym.compute_rhythm_metrics(beats)

    rules = RuleBasedReasoningEngine()
    rhythm = rules.classify_rhythm(measurements, rhythm_metrics, beats)
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

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
