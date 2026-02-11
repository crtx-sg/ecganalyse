# Project Context

## Purpose
AI-powered ECG interpretation system that analyzes raw electrocardiogram signals from clinical alarm events stored in HDF5 format. Processes 12-second ECG strips (6s before/after alarm) at 200Hz, extracts morphological features, calculates clinical measurements, detects abnormalities, and answers clinical queries through a Feature-Augmented LLM interface.

## Tech Stack
- Python 3.11+
- PyTorch 2.0+ (deep learning framework)
- h5py (HDF5 file I/O)
- FastAPI + Uvicorn (API layer)
- Pydantic 2.0+ (validation/schemas)
- NumPy, SciPy, NeuroKit2 (signal processing)
- mamba-ssm (state space models)
- torch-geometric (graph neural networks)
- Anthropic/OpenAI SDK (LLM integration)
- pytest (testing)

## Project Conventions

### Code Style
- PEP 8 with 100-char line limit
- Type hints on all public interfaces
- Dataclasses for data transfer objects
- f-strings for formatting

### Architecture Patterns
- Pipeline architecture: sequential stages with well-defined interfaces
- Neuro-symbolic: neural feature extraction + symbolic rule-based reasoning
- Feature-Augmented LLM: structured JSON features only (no raw signals to LLM)
- Each pipeline stage has its own module under `src/`

### Testing Strategy
- Static analysis: mypy, ruff
- Unit tests per module with pytest
- Integration tests per pipeline phase
- Gate tests (input/output contract validation) at phase boundaries
- CLI test harness for each phase enabling manual verification

### Git Workflow
- Main branch for stable releases
- Feature branches per phase or task group
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`

## Domain Context
- ECG = Electrocardiogram; measures heart electrical activity via surface leads
- 7-lead configuration: ECG1 (Lead I), ECG2 (Lead II), ECG3 (Lead III), aVR, aVL, aVF, vVX (chest)
- Fiducial points: P-wave onset/peak/offset, QRS onset/R-peak/offset, T-wave onset/peak/offset
- Clinical intervals: PR (atrial-ventricular delay), QRS (ventricular depolarization), QT/QTc (repolarization)
- Alarm events: clinical monitor alarms with contextual vitals (HR, SpO2, BP, RespRate, Temp)
- HDF5 files contain multiple alarm events per patient per month

## Important Constraints
- Feature-Augmented LLM ONLY: no raw signal tokens sent to LLM
- All measurements MUST have auditable calculation traces (regulatory: IEC 62304)
- Input exclusively from HDF5 alarm event format
- 7-lead ECG only (not standard 12-lead)
- Neuro-symbolic architecture required for explainability
- Alarm event context (event ID, timestamp, offset) must be preserved throughout pipeline

## External Dependencies
- HDF5 alarm event files from clinical monitoring devices (RMSAI-SimDevice-v1.0)
- LLM API (Anthropic Claude or OpenAI GPT) for clinical query answering
- No external databases required; all data from HDF5 files
