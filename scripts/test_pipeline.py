#!/usr/bin/env python3
"""Batch test: run inference on all test condition files and compare results.

Runs the full pipeline on each HDF5 file in data/test_conditions/ and prints
a summary table comparing predicted vs expected condition, HR, and beat count.

Usage:
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --test-dir data/test_conditions
    python scripts/test_pipeline.py --no-beats   # classification only
"""

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import PipelineConfig
from scripts.run_inference import run_inference

# Map filename label prefix → expected condition name
LABEL_TO_CONDITION = {
    "sinus_brady": "SINUS_BRADYCARDIA",
    "normal_sinus": "NORMAL_SINUS",
    "sinus_tachy": "SINUS_TACHYCARDIA",
    "afib": "ATRIAL_FIBRILLATION",
    "aflutter": "ATRIAL_FLUTTER",
    "svt": "SVT",
    "pvc": "PVC",
    "vtach": "VENTRICULAR_TACHYCARDIA",
    "lbbb": "LBBB",
    "rbbb": "RBBB",
    "avblock1": "AV_BLOCK_1",
    "avblock2t1": "AV_BLOCK_2_TYPE1",
    "ste": "ST_ELEVATION",
}


def parse_filename(name):
    """Extract expected condition and HR from filename like 'afib_120_medium.h5'.

    Returns:
        (expected_condition, expected_hr) or (None, None) if unparseable.
    """
    stem = name.replace(".h5", "")
    # Try each label prefix (longest first to avoid partial matches)
    for label in sorted(LABEL_TO_CONDITION, key=len, reverse=True):
        if stem.startswith(label + "_"):
            rest = stem[len(label) + 1:]
            # Extract HR number from remainder (e.g. "120_medium" -> 120)
            match = re.match(r"(\d+)", rest)
            if match:
                return LABEL_TO_CONDITION[label], float(match.group(1))
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Batch test across cardiac conditions")
    parser.add_argument("--test-dir", default="data/test_conditions",
                        help="Directory with test HDF5 files")
    parser.add_argument("--no-beats", action="store_true",
                        help="Skip beat detection (classification only)")
    args = parser.parse_args()

    if not os.path.isdir(args.test_dir):
        print(f"Error: directory not found: {args.test_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(f for f in os.listdir(args.test_dir) if f.endswith(".h5"))
    if not files:
        print(f"No .h5 files found in {args.test_dir}", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig(
        enable_beat_analysis=not args.no_beats,
        enable_heart_rate=not args.no_beats,
        enable_interval_measurements=not args.no_beats,
        beat_detector="signal",
    )

    # Header
    if args.no_beats:
        header = f"{'Condition':<24s} {'Pred Condition':<28s} {'Conf':>6s} {'Match':>5s}"
        sep = "-" * len(header)
    else:
        header = (
            f"{'Condition':<24s} {'Exp HR':>7s} {'Pred HR':>8s} {'HR Err':>7s} "
            f"{'Pred Condition':<28s} {'Conf':>6s} {'Beats':>5s} {'Match':>5s}"
        )
        sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    total = 0
    matches = 0
    hr_errors = []
    start_all = time.monotonic()

    for fname in files:
        fpath = os.path.join(args.test_dir, fname)
        expected_cond, expected_hr = parse_filename(fname)
        if expected_cond is None:
            print(f"  SKIP {fname} (cannot parse expected condition)")
            continue

        try:
            result = run_inference(fpath, "event_1001", config)
        except Exception as e:
            print(f"  FAIL {fname}: {e}")
            continue

        total += 1

        # Predicted condition
        cc = result.get("condition_classification", {})
        pred_cond = cc.get("condition", "?")
        conf = cc.get("confidence", 0.0)
        cond_match = pred_cond == expected_cond

        if cond_match:
            matches += 1
            mark = "Y"
        else:
            mark = "N"

        label = fname.replace("_medium.h5", "").replace("_clean.h5", "").replace(".h5", "")

        if args.no_beats:
            print(f"{label:<24s} {pred_cond:<28s} {conf:>6.2f} {mark:>5s}")
        else:
            gm = result.get("global_measurements")
            if gm is not None:
                pred_hr = gm.get("heart_rate_bpm", 0.0)
                hr_err = abs(pred_hr - expected_hr)
                hr_errors.append(hr_err)
                n_beats = len(result.get("beats", []))
                print(
                    f"{label:<24s} {expected_hr:>7.1f} {pred_hr:>8.1f} {hr_err:>7.1f} "
                    f"{pred_cond:<28s} {conf:>6.2f} {n_beats:>5d} {mark:>5s}"
                )
            else:
                print(
                    f"{label:<24s} {expected_hr:>7.1f} {'N/A':>8s} {'N/A':>7s} "
                    f"{pred_cond:<28s} {conf:>6.2f} {'N/A':>5s} {mark:>5s}"
                )

    elapsed = time.monotonic() - start_all
    print(sep)

    # Summary
    acc = matches / total * 100 if total > 0 else 0
    print(f"\nCondition accuracy: {matches}/{total} ({acc:.0f}%)")
    if hr_errors:
        import numpy as np
        mean_err = float(np.mean(hr_errors))
        max_err = float(np.max(hr_errors))
        print(f"HR error: mean={mean_err:.1f} bpm, max={max_err:.1f} bpm")
    print(f"Total time: {elapsed:.1f}s ({elapsed/max(total,1):.1f}s/event)")


if __name__ == "__main__":
    main()
