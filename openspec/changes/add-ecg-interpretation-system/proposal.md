# Change: Add AI-Powered ECG Interpretation System

## Why
Clinical monitoring systems generate high volumes of ECG alarm events stored in HDF5 files. Clinicians need automated, explainable ECG interpretation with contextual vitals integration and the ability to ask natural language clinical queries. No such system currently exists for this HDF5 alarm event format with a Feature-Augmented LLM approach.

## What Changes
- **Add data loading capability**: HDF5 file parser, alarm event parser, vitals parser, PyTorch dataset
- **Add signal preprocessing capability**: Signal quality assessment (SQI) and learned U-Net denoiser for 7-lead ECG
- **Add feature encoding capability**: ECG-TransCovNet (CNN + Transformer encoder-decoder) for condition classification; ECG-Mamba and 1D-Swin Transformer with dual-path fusion as alternative encoders
- **Add dense prediction capability**: Signal-based beat detector (default, classical DSP); U-Net heatmap decoder + GAT for multi-lead fusion (opt-in, requires training)
- **Add neural condition classification**: 16-class cardiac condition classifier using trained ECG-TransCovNet model
- **Add neuro-symbolic interpretation capability**: Symbolic calculation engine, rule-based reasoning, vitals context integration, JSON feature assembly with PipelineConfig for selective enable/disable
- **Add clinical query interface capability**: Query router, deterministic handler (Type A), Feature-Augmented LLM handler (Type B)
- **Add API service capability**: FastAPI application with endpoints for analysis and queries
- **Add unified inference scripts**: `run_inference.py` (single/batch inference with CLI flags), `test_pipeline.py` (batch validation across 13 conditions)
- **Add ECG simulator**: 16-condition ECG signal simulator for test data generation
- **Add CLI test harnesses**: Per-phase command-line tools for manual input/output verification
- **Add comprehensive test suite**: Static, gate, unit, and integration tests for every phase (351 tests passing)

## Impact
- Affected specs: All new — `data-loading`, `signal-preprocessing`, `feature-encoding`, `dense-prediction`, `neuro-symbolic-interpretation`, `clinical-query-interface`, `api-service`
- Affected code: Entire `src/` directory (new project)
- New dependencies: PyTorch, h5py, SciPy, FastAPI, Anthropic/OpenAI SDK
- New project structure: `config/`, `src/`, `tests/`, `scripts/`, `models/`, `data/`

## Phased Delivery
The system is delivered in 7 phases, each independently testable via CLI:

| Phase | Capability | CLI Entry Point | Status |
|-------|-----------|-----------------|--------|
| 0 | Data Loading | `python scripts/cli_phase0.py <hdf5_file> [event_id]` | Done |
| 1 | Signal Preprocessing | `python scripts/cli_phase1.py <hdf5_file> <event_id>` | Done |
| 2 | Feature Encoding | `python scripts/cli_phase2.py <hdf5_file> <event_id>` | Done |
| 3 | Dense Prediction (beat detection) | `python scripts/cli_phase3.py <hdf5_file> <event_id>` | Done |
| 3b | Condition Classification | via `run_inference.py` or `cli_phase4.py` | Done |
| 4 | Neuro-Symbolic Interpretation | `python scripts/cli_phase4.py <hdf5_file> <event_id>` | Done |
| 5 | Clinical Query Interface | `python scripts/cli_phase5.py <hdf5_file> <event_id> "<query>"` | Pending |
| 6 | API Service | `uvicorn src.api.app:app` | Pending |

### Unified Inference (recommended entry point)
```bash
python scripts/run_inference.py <hdf5_file> <event_id> [--no-beats] [--no-hr] [--no-intervals] [--all] [-o file.json]
python scripts/test_pipeline.py [--test-dir dir] [--no-beats]
```

## Current Architecture Notes

### Default Pipeline (as of latest)
- **Condition classification**: ECG-TransCovNet from `models/ecg_transcovnet/best_model.pt` (trained, 69% accuracy on 13 test conditions)
- **Beat detection**: `SignalBasedBeatDetector` (classical DSP — bandpass filter + adaptive R-peak detection + fiducial estimation). Mean HR error: 1.0 bpm across test conditions
- **Neural heatmap decoder**: Available but untrained (opt-in via `--heatmap` flag)
- **PipelineConfig**: Controls beat analysis, HR, and interval measurement stages independently

### ECG Simulator
A 16-condition ECG simulator (`src/simulator/`) generates realistic synthetic alarm events for testing and training. 13 test condition files are stored in `data/test_conditions/`.
