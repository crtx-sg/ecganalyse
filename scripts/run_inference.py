#!/usr/bin/env python3
"""Unified ECG inference pipeline: HDF5 -> full interpretation -> JSON.

Defaults to:
- TransCovNet condition classifier (models/ecg_transcovnet/best_model.pt)
- Signal-based beat detection (classical DSP, no neural heatmap model needed)
- Full measurements enabled

Usage:
    # Single event
    python scripts/run_inference.py data/test_conditions/afib_120_medium.h5 event_1001

    # Skip beat analysis (classification only)
    python scripts/run_inference.py data/test_conditions/afib_120_medium.h5 event_1001 --no-beats

    # Skip HR / intervals separately
    python scripts/run_inference.py data/test_conditions/afib_120_medium.h5 event_1001 --no-hr --no-intervals

    # All events in a file
    python scripts/run_inference.py data/samples/PT9401_2026-02.h5 --all

    # Output JSON to file
    python scripts/run_inference.py data/test_conditions/afib_120_medium.h5 event_1001 -o result.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import PipelineConfig
from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.quality import SignalQualityAssessor
from src.prediction.signal_beat_detector import SignalBasedBeatDetector
from src.prediction.condition_classifier import ConditionClassifier
from src.interpretation.symbolic import SymbolicCalculationEngine
from src.interpretation.rules import RuleBasedReasoningEngine
from src.interpretation.vitals_context import VitalsContextIntegrator
from src.interpretation.assembly import JSONAssembler


def run_inference(hdf5_path, event_id, config):
    """Run full inference pipeline on a single event.

    Args:
        hdf5_path: Path to HDF5 file.
        event_id: Event ID (e.g. 'event_1001').
        config: PipelineConfig controlling which stages run.

    Returns:
        JSON-serializable result dict.
    """
    start = time.monotonic()

    # Phase 0: Load
    loader = HDF5AlarmEventLoader()
    with loader.load_file(hdf5_path) as f:
        event = loader.load_event(f, event_id)

    # Phase 1: Quality
    assessor = SignalQualityAssessor()
    quality = assessor.assess(event.ecg)

    ecg_np = event.ecg.as_array
    ecg_tensor = torch.from_numpy(ecg_np).unsqueeze(0).float()

    # Phase 3a: Beat detection (signal-based by default)
    beats = []
    if config.enable_beat_analysis:
        detector = SignalBasedBeatDetector(fs=200)
        beats = detector.detect(ecg_np)

    # Phase 3b: Condition classification (always runs)
    condition_dict = None
    condition_pred = None
    try:
        classifier = ConditionClassifier()
        condition_pred = classifier.classify(ecg_tensor)
        condition_dict = {
            "condition": condition_pred.condition,
            "rhythm_label": condition_pred.rhythm_label,
            "confidence": condition_pred.confidence,
        }
    except FileNotFoundError:
        pass

    # Phase 4: Interpretation
    sym = SymbolicCalculationEngine(
        fs=200,
        enable_heart_rate=config.enable_heart_rate,
        enable_intervals=config.enable_interval_measurements,
    )

    measurements = None
    if config.enable_beat_analysis:
        measurements = sym.compute_global_measurements(beats)
    else:
        sym.traces.append("Beat analysis: disabled by pipeline config")

    rhythm_metrics = sym.compute_rhythm_metrics(beats)

    rules = RuleBasedReasoningEngine()
    rhythm = rules.classify_rhythm(
        measurements, rhythm_metrics, beats,
        condition_prediction=condition_dict,
    )
    findings = rules.generate_findings(measurements, rhythm, beats)

    vitals_integrator = VitalsContextIntegrator()
    if measurements is not None:
        vital_findings = vitals_integrator.integrate(measurements, event.vitals)
        findings.extend(vital_findings)

    # Assemble JSON
    assembler = JSONAssembler()
    result = assembler.assemble(
        event, quality, measurements, rhythm, beats,
        findings, sym.traces,
        processing_start_time=start,
        pipeline_config=config,
    )

    # Inject condition classification
    if condition_pred is not None:
        result["condition_classification"] = {
            "condition": condition_pred.condition,
            "rhythm_label": condition_pred.rhythm_label,
            "confidence": condition_pred.confidence,
            "top_predictions": [
                {"condition": name, "confidence": conf}
                for name, conf in condition_pred.top_k
            ],
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="ECG inference pipeline (TransCovNet + signal-based beats)",
    )
    parser.add_argument("hdf5_file", help="Path to HDF5 file")
    parser.add_argument("event_id", nargs="?", default=None,
                        help="Event ID (e.g. event_1001). Required unless --all.")
    parser.add_argument("--all", action="store_true",
                        help="Process all events in the file")
    parser.add_argument("--no-beats", action="store_true",
                        help="Skip beat detection (classification only)")
    parser.add_argument("--no-hr", action="store_true",
                        help="Skip heart rate calculation")
    parser.add_argument("--no-intervals", action="store_true",
                        help="Skip interval measurements (PR, QRS, QT)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write JSON output to file instead of stdout")
    args = parser.parse_args()

    if not args.all and args.event_id is None:
        parser.error("event_id is required unless --all is specified")

    # Build PipelineConfig from CLI flags
    config = PipelineConfig(
        enable_beat_analysis=not args.no_beats,
        enable_heart_rate=not args.no_hr,
        enable_interval_measurements=not args.no_intervals,
        beat_detector="signal",
    )

    # Determine which events to process
    loader = HDF5AlarmEventLoader()
    if args.all:
        with loader.load_file(args.hdf5_file) as f:
            event_ids = loader.list_events(f)
        if not event_ids:
            print("No events found in file.", file=sys.stderr)
            sys.exit(1)
        print(f"Processing {len(event_ids)} events from {args.hdf5_file}...",
              file=sys.stderr)
    else:
        event_ids = [args.event_id]

    # Run inference
    results = []
    for eid in event_ids:
        if len(event_ids) > 1:
            print(f"  {eid}...", file=sys.stderr)
        result = run_inference(args.hdf5_file, eid, config)
        results.append(result)

    # Output
    output = results[0] if len(results) == 1 else results
    json_str = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str + "\n")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
