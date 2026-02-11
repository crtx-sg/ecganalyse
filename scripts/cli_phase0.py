"""CLI harness for Phase 0: Data Loading.

Usage:
    python scripts/cli_phase0.py <hdf5_file> [event_id]

Without event_id: lists all events and file metadata.
With event_id: prints full event data summary as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Ensure project root is on sys.path so src/config imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.data.vitals_parser import VitalsParser
from src.ecg_system.exceptions import ECGSystemError


def _ecg_summary(event: Any) -> dict[str, Any]:
    """Summarize ECG signal statistics."""
    summary: dict[str, Any] = {
        "leads": event.ecg.leads,
        "num_leads": len(event.ecg.leads),
        "sample_rate": event.ecg.sample_rate,
        "num_samples": event.ecg.num_samples,
        "duration_sec": event.ecg.duration_sec,
        "pacer_info": event.ecg.extras.get("pacer_info", 0),
        "per_lead": {},
    }
    for lead in event.ecg.leads:
        sig = event.ecg.signals[lead]
        summary["per_lead"][lead] = {
            "min": round(float(np.min(sig)), 4),
            "max": round(float(np.max(sig)), 4),
            "mean": round(float(np.mean(sig)), 4),
            "std": round(float(np.std(sig)), 4),
            "samples": int(sig.shape[0]),
        }
    return summary


def _aux_summary(event: Any) -> dict[str, Any]:
    """Summarize auxiliary signals."""
    result: dict[str, Any] = {}
    if event.ppg is not None:
        result["ppg"] = {
            "sample_rate": event.ppg.sample_rate,
            "samples": int(event.ppg.signal.shape[0]),
            "min": round(float(np.min(event.ppg.signal)), 4),
            "max": round(float(np.max(event.ppg.signal)), 4),
        }
    if event.resp is not None:
        result["resp"] = {
            "sample_rate": event.resp.sample_rate,
            "samples": int(event.resp.signal.shape[0]),
            "min": round(float(np.min(event.resp.signal)), 4),
            "max": round(float(np.max(event.resp.signal)), 4),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0: Load and inspect HDF5 alarm event data"
    )
    parser.add_argument("hdf5_file", help="Path to HDF5 alarm event file")
    parser.add_argument("event_id", nargs="?", help="Event ID (e.g., event_1001)")
    args = parser.parse_args()

    loader = HDF5AlarmEventLoader()

    try:
        with loader.load_file(args.hdf5_file) as f:
            metadata = loader.load_metadata(f)

            if args.event_id is None:
                # List mode: show file metadata and all events
                events = loader.list_events(f)
                output = {
                    "file": args.hdf5_file,
                    "file_metadata": {
                        "patient_id": metadata.patient_id,
                        "sampling_rate_ecg": metadata.sampling_rate_ecg,
                        "sampling_rate_ppg": metadata.sampling_rate_ppg,
                        "sampling_rate_resp": metadata.sampling_rate_resp,
                        "alarm_offset_seconds": metadata.alarm_offset_seconds,
                        "data_quality_score": metadata.data_quality_score,
                        "device_info": metadata.device_info,
                        "max_vital_history": metadata.max_vital_history,
                    },
                    "events": events,
                    "num_events": len(events),
                }
            else:
                # Detail mode: show full event summary
                event = loader.load_event(f, args.event_id)
                vitals_parser = VitalsParser()

                vitals_summary: dict[str, Any] = {}
                threshold_violations: list[dict[str, Any]] = []
                if event.vitals is not None:
                    vitals_summary = event.vitals.to_dict()
                    violations = vitals_parser.check_threshold_violations(event.vitals)
                    threshold_violations = [
                        {
                            "vital": v.vital_name,
                            "value": v.value,
                            "units": v.units,
                            "type": v.threshold_type,
                            "threshold": v.threshold_value,
                        }
                        for v in violations
                    ]

                output = {
                    "event_id": event.event_id,
                    "uuid": event.uuid,
                    "timestamp": event.timestamp,
                    "metadata": {
                        "patient_id": metadata.patient_id,
                        "device_info": metadata.device_info,
                        "alarm_offset_seconds": metadata.alarm_offset_seconds,
                        "data_quality_score": metadata.data_quality_score,
                    },
                    "ecg_summary": _ecg_summary(event),
                    "auxiliary_summary": _aux_summary(event),
                    "vitals_summary": vitals_summary,
                    "threshold_violations": threshold_violations,
                }

        print(json.dumps(output, indent=2))

    except ECGSystemError as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
