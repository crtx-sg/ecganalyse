#!/usr/bin/env python3
"""Visualize Phase 4 (Neuro-Symbolic Interpretation) output.

Runs full pipeline Phase 0–4 on an HDF5 event and produces:
  1. Multi-panel clinical summary figure (PNG)
  2. JSON Feature Assembly printed to stdout (optional)

Usage:
    python scripts/visualize_phase4.py data/samples/PT1234_2024-01.h5 event_1001
    python scripts/visualize_phase4.py data/samples/PT9401_2026-02.h5 event_1001 --output afib_report.png
    python scripts/visualize_phase4.py data/samples/PT1234_2024-01.h5 event_1001 --json
    python scripts/visualize_phase4.py data/samples/PT1234_2024-01.h5 event_1001 --all-leads
"""

import argparse
import json
import os
import sys
import time
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
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


LEAD_ORDER = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]

SEVERITY_COLORS = {
    "normal": "#27ae60",
    "mild": "#f39c12",
    "moderate": "#e67e22",
    "severe": "#e74c3c",
    "critical": "#8e44ad",
}

SEVERITY_EMOJI = {
    "normal": "OK",
    "mild": "(!)",
    "moderate": "(!!)",
    "severe": "(!!!)",
    "critical": "CRIT",
}


def run_pipeline(hdf5_path, event_id, signal_based=False):
    """Run Phase 0-4 and return (assembly_dict, event, beats, heatmaps_np, ecg_np).

    Args:
        signal_based: If True, skip neural Phase 2-3 and use direct signal
                      analysis for beat detection. Recommended for clean data
                      or when neural models are untrained.
    """
    start = time.monotonic()

    loader = HDF5AlarmEventLoader()
    with loader.load_file(hdf5_path) as f:
        event = loader.load_event(f, event_id)

    # Phase 1
    assessor = SignalQualityAssessor()
    quality = assessor.assess(event.ecg)

    ecg_np = event.ecg.as_array
    hm_np = None

    if signal_based:
        # Direct signal-based beat detection (bypasses untrained neural models)
        detector = SignalBasedBeatDetector(fs=200)
        beats = detector.detect(ecg_np)
    else:
        denoiser = ECGDenoiser()
        denoiser.eval()
        ecg_tensor = torch.from_numpy(ecg_np).unsqueeze(0).float()
        with torch.no_grad():
            denoised = denoiser(ecg_tensor)

        # Phase 2
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()
        with torch.no_grad():
            features = encoder(denoised)

        # Phase 3
        decoder = HeatmapDecoder(d_model=256)
        decoder.eval()
        with torch.no_grad():
            heatmaps = decoder(features)

        hm_np = heatmaps.squeeze(0).numpy()
        extractor = FiducialExtractor()
        beats = extractor.extract(hm_np, ecg_np)

    # Phase 4
    sym = SymbolicCalculationEngine(fs=200)
    measurements = sym.compute_global_measurements(beats)
    rhythm_metrics = sym.compute_rhythm_metrics(beats)

    rules = RuleBasedReasoningEngine()
    rhythm = rules.classify_rhythm(measurements, rhythm_metrics, beats)
    findings = rules.generate_findings(measurements, rhythm, beats)

    vitals_integrator = VitalsContextIntegrator()
    vital_findings = vitals_integrator.integrate(measurements, event.vitals)
    findings.extend(vital_findings)

    assembler = JSONAssembler()
    result = assembler.assemble(
        event, quality, measurements, rhythm, beats,
        findings, sym.traces, processing_start_time=start,
    )

    return result, event, beats, hm_np, ecg_np


def plot_report(result, event, beats, hm_np, ecg_np, leads_to_show, output_path):
    """Create a multi-panel clinical report figure."""
    fs = 200
    n_leads = len(leads_to_show)
    lead_indices = [LEAD_ORDER.index(l) for l in leads_to_show]

    # Layout: top row = ECG strips with fiducials, bottom left = findings table,
    # bottom right = measurements + rhythm + vitals
    n_rows_ecg = n_leads
    total_rows = n_rows_ecg + 4  # ECG rows + summary rows

    fig = plt.figure(figsize=(18, 2.2 * n_rows_ecg + 8), facecolor="white")
    gs = gridspec.GridSpec(
        total_rows, 2,
        height_ratios=[1.0] * n_rows_ecg + [0.6, 1.5, 1.5, 1.0],
        width_ratios=[3, 2],
        hspace=0.35, wspace=0.3,
    )

    # --- Title bar ---
    ctx = result["event_context"]
    rhythm_clf = result["rhythm"]["classification"].replace("_", " ").title()
    hr = result["global_measurements"]["heart_rate_bpm"]
    patient = ctx.get("patient_id", "Unknown")
    sqi = result["quality"]["overall_sqi"]

    title = (
        f"ECG Interpretation Report  |  {patient}  |  {ctx['event_id']}  |  "
        f"{rhythm_clf}  |  HR {hr:.0f} bpm  |  SQI {sqi:.2f}"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995, color="#2c3e50")

    t = np.arange(ecg_np.shape[1]) / fs

    # --- ECG strips with fiducial markers ---
    fiducial_colors = {
        "P_onset": "#3498db", "P_peak": "#2980b9", "P_offset": "#3498db",
        "QRS_onset": "#e74c3c", "R_peak": "#c0392b", "QRS_offset": "#e74c3c",
        "T_onset": "#27ae60", "T_peak": "#229954", "T_offset": "#27ae60",
    }
    fiducial_markers = {
        "P_onset": "v", "P_peak": "^", "P_offset": "v",
        "QRS_onset": "|", "R_peak": "D", "QRS_offset": "|",
        "T_onset": "s", "T_peak": "o", "T_offset": "s",
    }

    ecg_axes = []
    for row_idx, (lead_name, lead_i) in enumerate(zip(leads_to_show, lead_indices)):
        ax = fig.add_subplot(gs[row_idx, :])
        ecg_axes.append(ax)
        signal = ecg_np[lead_i]
        ax.plot(t, signal, linewidth=0.7, color="#2c3e50", alpha=0.85)
        ax.set_ylabel(lead_name, fontsize=10, fontweight="bold", rotation=0, labelpad=40)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_facecolor("#fafafa")
        ax.tick_params(labelsize=8)

        # Plot fiducial points for beats on this lead
        for beat in beats:
            for fid_name, fp in beat.fiducials.items():
                if fp.sample < len(signal):
                    color = fiducial_colors.get(fid_name, "#888")
                    marker = fiducial_markers.get(fid_name, "o")
                    ax.plot(
                        fp.sample / fs, signal[fp.sample],
                        marker=marker, color=color, markersize=5,
                        alpha=0.7, zorder=5,
                    )

        # Mark beat boundaries with light vertical lines at R-peaks
        for beat in beats:
            if "R_peak" in beat.fiducials:
                r_t = beat.fiducials["R_peak"].sample / fs
                ax.axvline(r_t, color="#c0392b", alpha=0.15, linewidth=0.8, linestyle=":")

    ecg_axes[-1].set_xlabel("Time (seconds)", fontsize=10)

    # Add fiducial legend to first ECG axis
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="^", color="#2980b9", linestyle="None", markersize=6, label="P-wave"),
        Line2D([0], [0], marker="D", color="#c0392b", linestyle="None", markersize=6, label="R-peak"),
        Line2D([0], [0], marker="o", color="#229954", linestyle="None", markersize=6, label="T-wave"),
    ]
    ecg_axes[0].legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.8)

    # --- Rhythm & Measurements panel (bottom-left, row n_rows_ecg) ---
    ax_header = fig.add_subplot(gs[n_rows_ecg, :])
    ax_header.axis("off")
    ax_header.text(0.0, 0.5, "INTERPRETATION SUMMARY", fontsize=12, fontweight="bold",
                   transform=ax_header.transAxes, va="center", color="#2c3e50")
    ax_header.axhline(y=0.1, xmin=0, xmax=1, color="#bdc3c7", linewidth=1.5)

    # --- Measurements panel ---
    ax_meas = fig.add_subplot(gs[n_rows_ecg + 1, 0])
    ax_meas.axis("off")

    gm = result["global_measurements"]
    rr = gm["rr_intervals_ms"]
    rhythm_info = result["rhythm"]

    meas_lines = [
        f"Heart Rate:  {gm['heart_rate_bpm']:.1f} bpm",
        f"RR Mean:  {rr['mean']:.0f} ms  (std {rr['std']:.0f}, range {rr['min']:.0f}-{rr['max']:.0f})",
        f"QRS Duration:  {gm['qrs_duration_ms']['median']:.0f} ms",
        f"QT Interval:  {gm['qt_interval_ms']:.0f} ms",
        f"QTc Bazett:  {gm['qtc_bazett_ms']:.1f} ms",
        f"QTc Fridericia:  {gm['qtc_fridericia_ms']:.1f} ms",
    ]
    pr = gm.get("pr_interval_ms")
    if pr and isinstance(pr, dict):
        meas_lines.insert(2, f"PR Interval:  {pr['median']:.0f} ms")
    elif pr is not None:
        meas_lines.insert(2, f"PR Interval:  {pr:.0f} ms")

    for i, line in enumerate(meas_lines):
        y = 0.92 - i * 0.13
        ax_meas.text(0.02, y, line, fontsize=9.5, fontfamily="monospace",
                     transform=ax_meas.transAxes, va="top", color="#2c3e50")

    ax_meas.text(0.02, 1.0, "MEASUREMENTS", fontsize=10, fontweight="bold",
                 transform=ax_meas.transAxes, va="top", color="#34495e")

    # --- Rhythm panel ---
    ax_rhythm = fig.add_subplot(gs[n_rows_ecg + 1, 1])
    ax_rhythm.axis("off")

    clf_label = rhythm_info["classification"].replace("_", " ").title()
    conf = rhythm_info["classification_confidence"]
    reg = rhythm_info["regularity"]
    p_morph = rhythm_info["p_wave_morphology"]
    p_ratio = rhythm_info["p_wave_presence_ratio"]
    ect = rhythm_info["ectopic_beats"]

    rhythm_lines = [
        f"Classification:  {clf_label}",
        f"Confidence:  {conf:.0%}",
        f"Regularity:  {reg} ({rhythm_info['regularity_score']:.2f})",
        f"P-wave:  {p_morph} (ratio {p_ratio:.2f})",
        f"P-QRS:  {rhythm_info['p_qrs_relationship']}",
        f"Ectopics:  PVC {ect['pvc_count']}, PAC {ect['pac_count']} / {ect['total_beats']} beats",
    ]

    ax_rhythm.text(0.02, 1.0, "RHYTHM", fontsize=10, fontweight="bold",
                   transform=ax_rhythm.transAxes, va="top", color="#34495e")
    for i, line in enumerate(rhythm_lines):
        y = 0.92 - i * 0.13
        ax_rhythm.text(0.02, y, line, fontsize=9.5, fontfamily="monospace",
                       transform=ax_rhythm.transAxes, va="top", color="#2c3e50")

    # --- Findings panel ---
    ax_find = fig.add_subplot(gs[n_rows_ecg + 2, 0])
    ax_find.axis("off")

    ax_find.text(0.02, 1.0, "FINDINGS", fontsize=10, fontweight="bold",
                 transform=ax_find.transAxes, va="top", color="#34495e")

    findings = result["findings"]
    for i, f in enumerate(findings[:8]):
        y = 0.88 - i * 0.11
        sev = f["severity"]
        tag = SEVERITY_EMOJI.get(sev, "")
        color = SEVERITY_COLORS.get(sev, "#666")
        text = f"[{sev.upper():>8s}] {f['finding']}"
        if len(text) > 70:
            text = text[:67] + "..."
        ax_find.text(0.02, y, text, fontsize=8.5, fontfamily="monospace",
                     transform=ax_find.transAxes, va="top", color=color)

    if len(findings) > 8:
        ax_find.text(0.02, 0.88 - 8 * 0.11, f"  ... and {len(findings) - 8} more",
                     fontsize=8.5, fontfamily="monospace", transform=ax_find.transAxes,
                     va="top", color="#999")

    # --- Vitals panel ---
    ax_vitals = fig.add_subplot(gs[n_rows_ecg + 2, 1])
    ax_vitals.axis("off")

    ax_vitals.text(0.02, 1.0, "VITALS CONTEXT", fontsize=10, fontweight="bold",
                   transform=ax_vitals.transAxes, va="top", color="#34495e")

    vc = result.get("vitals_context")
    if vc:
        vital_order = ["hr", "pulse", "spo2", "systolic", "diastolic", "resp_rate", "temp"]
        row = 0
        for vname in vital_order:
            if vname in vc:
                v = vc[vname]
                val = v["value"]
                units = v["units"]
                violation = v.get("threshold_violation", False)
                color = "#e74c3c" if violation else "#2c3e50"
                flag = " *" if violation else ""
                y = 0.88 - row * 0.12
                ax_vitals.text(0.02, y, f"{vname:>12s}: {val:>6.1f} {units}{flag}",
                               fontsize=9, fontfamily="monospace",
                               transform=ax_vitals.transAxes, va="top", color=color)
                row += 1

        # HR validation
        hrv = gm.get("heart_rate_validation")
        if hrv:
            y = 0.88 - row * 0.12
            status = hrv["status"].upper()
            diff = hrv["difference_bpm"]
            color = "#27ae60" if status == "CONSISTENT" else "#e74c3c"
            ax_vitals.text(0.02, y, f"  HR valid.: {status} (diff {diff:.1f})",
                           fontsize=9, fontfamily="monospace",
                           transform=ax_vitals.transAxes, va="top", color=color)
    else:
        ax_vitals.text(0.02, 0.85, "No vitals data available", fontsize=9,
                       fontfamily="monospace", transform=ax_vitals.transAxes,
                       va="top", color="#999")

    # --- Summary bar ---
    ax_sum = fig.add_subplot(gs[n_rows_ecg + 3, :])
    ax_sum.axis("off")

    summary = result["summary"]
    interp = summary["primary_interpretation"]
    n_abn = summary["abnormality_count"]
    crits = summary["critical_findings"]

    bg_color = "#27ae60" if n_abn == 0 else ("#e67e22" if not crits else "#e74c3c")
    ax_sum.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.02",
        facecolor=bg_color, alpha=0.12,
        transform=ax_sum.transAxes,
    ))
    ax_sum.text(0.02, 0.7, f"SUMMARY: {interp}",
                fontsize=10, fontweight="bold", transform=ax_sum.transAxes,
                va="center", color="#2c3e50", wrap=True)
    cats = ", ".join(summary["categories_present"])
    ax_sum.text(0.02, 0.25, f"Categories: {cats}  |  Abnormalities: {n_abn}  |  "
                f"Processing: {result['processing_time_ms']:.0f} ms  |  "
                f"Schema: {result['schema_version']}",
                fontsize=8.5, transform=ax_sum.transAxes, va="center", color="#7f8c8d")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved report: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 interpretation visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python scripts/visualize_phase4.py data/samples/PT1234_2024-01.h5 event_1001
          python scripts/visualize_phase4.py data/samples/PT9401_2026-02.h5 event_1001 --all-leads
          python scripts/visualize_phase4.py data/samples/PT1234_2024-01.h5 event_1001 --json --output report.png
        """),
    )
    parser.add_argument("hdf5_file", help="Path to HDF5 file")
    parser.add_argument("event_id", help="Event ID (e.g. event_1001)")
    parser.add_argument("--leads", nargs="+", default=["ECG2"],
                        help="Leads to show (default: ECG2)")
    parser.add_argument("--all-leads", action="store_true",
                        help="Show all 7 leads")
    parser.add_argument("--output", default="phase4_report.png",
                        help="Output PNG path (default: phase4_report.png)")
    parser.add_argument("--json", action="store_true",
                        help="Also print full JSON assembly to stdout")
    parser.add_argument("--signal-based", action="store_true",
                        help="Use signal-based beat detection (bypasses neural Phase 2-3)")

    args = parser.parse_args()

    if args.all_leads:
        leads = LEAD_ORDER
    else:
        leads = args.leads

    mode = "signal-based" if args.signal_based else "neural (Phase 2-3)"
    print(f"Running Phase 0-4 pipeline on {args.hdf5_file} / {args.event_id} [{mode}] ...")
    result, event, beats, hm_np, ecg_np = run_pipeline(
        args.hdf5_file, args.event_id, signal_based=args.signal_based,
    )

    print(f"  Detected {len(beats)} beats")
    print(f"  Rhythm: {result['rhythm']['classification']}")
    print(f"  HR: {result['global_measurements']['heart_rate_bpm']:.1f} bpm")
    print(f"  Findings: {len(result['findings'])}")
    print(f"  Processing: {result['processing_time_ms']:.0f} ms")

    plot_report(result, event, beats, hm_np, ecg_np, leads, args.output)

    if args.json:
        print("\n" + "=" * 60)
        print("JSON FEATURE ASSEMBLY v1.1")
        print("=" * 60)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
