# Change: Add AI-Powered ECG Interpretation System

## Why
Clinical monitoring systems generate high volumes of ECG alarm events stored in HDF5 files. Clinicians need automated, explainable ECG interpretation with contextual vitals integration and the ability to ask natural language clinical queries. No such system currently exists for this HDF5 alarm event format with a Feature-Augmented LLM approach.

## What Changes
- **Add data loading capability**: HDF5 file parser, alarm event parser, vitals parser, PyTorch dataset
- **Add signal preprocessing capability**: Signal quality assessment (SQI) and learned U-Net denoiser for 7-lead ECG
- **Add feature encoding capability**: ECG-Mamba (global context) and 1D-Swin Transformer (local morphology) with dual-path fusion
- **Add dense prediction capability**: U-Net heatmap decoder, Graph Attention Network for 7-lead fusion, fiducial point extraction
- **Add neuro-symbolic interpretation capability**: Symbolic calculation engine, rule-based reasoning, vitals context integration, JSON feature assembly
- **Add clinical query interface capability**: Query router, deterministic handler (Type A), Feature-Augmented LLM handler (Type B)
- **Add API service capability**: FastAPI application with endpoints for analysis and queries
- **Add CLI test harnesses**: Per-phase command-line tools for manual input/output verification
- **Add comprehensive test suite**: Static, gate, unit, and integration tests for every phase

## Impact
- Affected specs: All new — `data-loading`, `signal-preprocessing`, `feature-encoding`, `dense-prediction`, `neuro-symbolic-interpretation`, `clinical-query-interface`, `api-service`
- Affected code: Entire `src/` directory (new project)
- New dependencies: PyTorch, h5py, mamba-ssm, torch-geometric, FastAPI, Anthropic/OpenAI SDK
- New project structure: `config/`, `src/`, `tests/`, `scripts/`, `models/`, `data/`, `docs/`, `docker/`

## Phased Delivery
The system is delivered in 7 phases, each independently testable via CLI:

| Phase | Capability | CLI Entry Point |
|-------|-----------|-----------------|
| 0 | Data Loading | `python -m scripts.cli_phase0 <hdf5_file> [event_id]` |
| 1 | Signal Preprocessing | `python -m scripts.cli_phase1 <hdf5_file> <event_id>` |
| 2 | Feature Encoding | `python -m scripts.cli_phase2 <hdf5_file> <event_id>` |
| 3 | Dense Prediction | `python -m scripts.cli_phase3 <hdf5_file> <event_id>` |
| 4 | Neuro-Symbolic Interpretation | `python -m scripts.cli_phase4 <hdf5_file> <event_id>` |
| 5 | Clinical Query Interface | `python -m scripts.cli_phase5 <hdf5_file> <event_id> "<query>"` |
| 6 | API Service | `uvicorn src.api.app:app` |

Each phase CLI prints structured JSON output and validates inputs/outputs against schemas.
