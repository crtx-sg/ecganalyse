## Context
This is a greenfield AI-powered ECG interpretation system processing clinical alarm events from HDF5 files. It must provide explainable, auditable ECG analysis through a neuro-symbolic pipeline and answer clinical queries via a Feature-Augmented LLM interface. The system is subject to IEC 62304 medical device software requirements.

**Stakeholders**: Clinical engineers, cardiologists, regulatory affairs, ML engineers.

**Constraints**:
- 7-lead ECG input only (not standard 12-lead)
- HDF5 alarm event format is the sole input source
- Feature-Augmented LLM: structured JSON only, no raw signals to LLM
- All measurements require auditable calculation traces
- Neuro-symbolic architecture for regulatory explainability

## Goals / Non-Goals

### Goals
- Build a complete end-to-end ECG interpretation pipeline from HDF5 input to clinical query responses
- Provide a CLI test harness at each pipeline phase for independent verification
- Comprehensive test coverage: static analysis, gate tests, unit tests, integration tests
- Clean separation of concerns across pipeline stages
- Auditable calculation traces on every derived measurement
- Contextual vitals integration alongside ECG analysis
- Configurable pipeline stages via PipelineConfig (enable/disable beats, HR, intervals independently)

### Non-Goals
- Real-time streaming ECG processing (batch/event-based only)
- 12-lead ECG support (7-lead only for this release)
- Training pipeline optimization (basic training scripts only)
- Mobile or embedded deployment
- Multi-patient concurrent processing optimization
- FDA/CE regulatory submission (documentation support only)

## Architecture Decisions

### Decision 1: Six-Stage Pipeline Architecture
**What**: Linear pipeline with well-defined interfaces between stages: Data Loading -> Preprocessing -> Encoding -> Prediction -> Interpretation -> Query Interface.
**Why**: Each stage is independently testable, replaceable, and maps cleanly to the regulatory V-model. Stages communicate through typed dataclasses, enabling contract-based testing (gate tests) at every boundary.
**Alternatives considered**:
- Monolithic model (end-to-end neural): Rejected — no explainability, no calculation traces
- Microservices per stage: Over-engineered for current scale; can be split later

### Decision 2: Neuro-Symbolic Dual Architecture
**What**: Neural networks for condition classification and optional beat detection; symbolic engine for measurement calculation and clinical rule application.
**Why**: Neural components handle pattern recognition in noisy signals; symbolic components provide deterministic, auditable measurements required by IEC 62304. This separation means the LLM never sees raw signals — only structured features.
**Alternatives considered**:
- Pure neural interpretation: No auditability; fails regulatory requirements
- Pure symbolic/rule-based: Insufficient for complex morphology detection in noisy real-world signals

### Decision 3: ECG-TransCovNet for Condition Classification
**What**: CNN backbone (with Selective Kernel convolutions) + Transformer encoder-decoder architecture for 16-class cardiac condition classification. Trained checkpoint at `models/ecg_transcovnet/best_model.pt`.
**Why**: TransCovNet captures both local morphological features via CNN and global temporal dependencies via Transformer attention. The DETR-style object query decoder enables per-class attention weights for interpretability. Achieves 69% accuracy across 13 test conditions on simulated data.
**Alternatives considered**:
- Dual-path Mamba + Swin: Available as alternative encoders in `src/encoding/` but not used for the default classification pipeline
- Single CNN: Limited temporal context
- Single Transformer: Computationally expensive on raw 2400-sample sequences without CNN down-sampling

### Decision 4: Signal-Based Beat Detection as Default
**What**: Classical DSP pipeline for beat detection: bandpass filtering (1-40 Hz) + adaptive amplitude-based R-peak detection + waveform morphology-based fiducial estimation. Implemented in `SignalBasedBeatDetector`.
**Why**: The neural heatmap decoder (Stage B/C) has not been trained — only Stage A denoiser weights exist. Signal-based detection produces reliable results (mean HR error: 1.0 bpm across test conditions) without requiring any trained neural model. The heatmap pipeline remains available via `--heatmap` opt-in flag for when the model is trained.
**Alternatives considered**:
- Neural heatmap decoder (default): Rejected — produces garbage fiducials with untrained weights
- NeuroKit2 / external libraries: Adds dependency; the custom detector is tailored to our 7-lead format and produces Beat/FiducialPoint objects directly

### Decision 5: PipelineConfig for Selective Stage Control
**What**: `PipelineConfig` dataclass with flags: `enable_beat_analysis`, `enable_heart_rate`, `enable_interval_measurements`, `beat_detector`. Exposed via CLI args (`--no-beats`, `--no-hr`, `--no-intervals`).
**Why**: Enables running classification-only mode (fast, no beat detection needed) or selectively disabling measurements that may be unreliable. Critical for deployment where different use cases need different pipeline depth.
**Current behavior**:
- `enable_beat_analysis=False`: Skips Phase 3 entirely, condition classification still runs
- `enable_heart_rate=False`: Beats detected but HR not computed from RR intervals
- `enable_interval_measurements=False`: Beats detected but PR/QRS/QT/QTc not computed

### Decision 6: Feature-Augmented LLM (Not Signal-Augmented)
**What**: The LLM receives only structured JSON (measurements, findings, vitals, traces) and clinical queries. Never raw signals, embeddings, or neural features.
**Why**: Regulatory requirement — every LLM response must be traceable to deterministic measurements. Also simpler to validate, test, and audit. LLM serves as a clinical reasoning and natural language interface, not a signal processor.
**Alternatives considered**:
- Signal tokens to LLM: No auditability; hallucination risk on measurements
- No LLM (rule-based NLG only): Limited expressiveness for complex clinical queries

### Decision 7: Per-Phase CLI Test Harness + Unified Inference Script
**What**: Each pipeline phase has a standalone CLI script (`scripts/cli_phaseN.py`). Additionally, `scripts/run_inference.py` provides a unified entry point that runs the recommended default pipeline (TransCovNet + signal-based beats) with PipelineConfig flags.
**Why**: Phase CLIs enable independent development and debugging. The unified script provides the recommended production-like entry point. `scripts/test_pipeline.py` runs batch validation across all 13 test conditions.
**Alternatives considered**:
- API-only testing: Requires full stack; too heavyweight for phase development
- Notebook-only testing: Not automatable in CI

### Decision 8: Gate Tests at Phase Boundaries
**What**: Pytest-based gate tests that validate the output schema and value ranges of each pipeline phase against its contract (input/output dataclass shapes, required fields, value bounds).
**Why**: Catches interface drift between phases early. If Phase 2's output shape changes, Phase 3's gate test fails immediately — before any logic errors cascade. Critical for multi-developer parallel work.
**Alternatives considered**:
- Integration tests only: Catch issues too late
- Manual contract review: Error-prone, doesn't scale

### Decision 9: ECG Simulator for Test Data
**What**: 16-condition ECG signal simulator (`src/simulator/`) that generates realistic synthetic HDF5 alarm events with configurable heart rate, noise level, and condition-specific morphology.
**Why**: Enables comprehensive testing without requiring real patient data. The simulator produces signals with known ground truth (condition, HR, fiducial locations) for validation. 13 test condition files in `data/test_conditions/` serve as the standard validation set.

## Data Flow

```
HDF5 File
    |
    v
+----------------------------------+
| Phase 0: Data Loading            |
|  Input:  HDF5 filepath + event   |
|  Output: AlarmEvent dataclass    |
|  CLI:    cli_phase0              |
|  Tests:  test_data/              |
+---------------+------------------+
                | AlarmEvent
                v
+----------------------------------+
| Phase 1: Signal Preprocessing    |
|  Input:  AlarmEvent.ecg (7,2400) |
|  Output: QualityReport +         |
|          denoised ECG (7,2400)   |
|  CLI:    cli_phase1              |
|  Tests:  test_preprocessing/     |
+---------------+------------------+
                |
       +--------+--------+
       |                  |
       v                  v
+----------------+  +-------------------+
| Phase 3a: Beat |  | Phase 3b:         |
| Detection      |  | Condition Classify |
| SignalBased    |  | TransCovNet (16cl) |
| BeatDetector   |  | ConditionClassifier|
| (default)      |  | (always runs)      |
+-------+--------+  +--------+----------+
        |                     |
        +----------+----------+
                   |
                   v
+----------------------------------+
| Phase 4: Interpretation          |
|  SymbolicCalculationEngine       |
|  RuleBasedReasoningEngine        |
|  VitalsContextIntegrator         |
|  JSONAssembler                   |
|  Output: JSON Feature Assembly   |
|  CLI:    cli_phase4              |
|  Tests:  test_interpretation/    |
+---------------+------------------+
                | Dict (JSON Assembly)
                v
+----------------------------------+
| Phase 5: Clinical Query          |
|  Input:  JSON Assembly + query   |
|  Output: natural language answer |
|  CLI:    cli_phase5              |
|  Status: Pending                 |
+---------------+------------------+
                |
                v
+----------------------------------+
| Phase 6: API Service             |
|  FastAPI wrapping full pipeline  |
|  CLI:    uvicorn                 |
|  Status: Pending                 |
+----------------------------------+
```

## Testing Strategy Per Phase

Each phase includes four test categories:

| Category | Purpose | Tool | When |
|----------|---------|------|------|
| **Static** | Type checking, linting | mypy, ruff | Pre-commit, CI |
| **Unit** | Individual function/class correctness | pytest | Every commit |
| **Gate** | Input/output contract validation at phase boundaries | pytest | Every commit |
| **Integration** | End-to-end within phase, cross-phase | pytest | PR merge, nightly |

### Current Test Coverage
- 351 unit/module tests passing
- 13-condition batch validation via `test_pipeline.py`
- Per-phase CLI harnesses for manual verification

### Gate Test Pattern
```python
# tests/gates/test_phase0_gate.py
def test_phase0_output_contract(sample_hdf5_file):
    """Gate test: Phase 0 output matches AlarmEvent contract."""
    loader = HDF5AlarmEventLoader()
    with loader.load_file(sample_hdf5_file) as f:
        event = loader.load_event(f, "event_1001")

    # Shape contracts
    assert event.ecg.as_array.shape == (7, 2400)
    assert event.ecg.sample_rate == 200
    assert set(event.ecg.leads) == {"ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"}

    # Value range contracts
    assert all(-10.0 <= event.ecg.signals[lead].max() <= 10.0 for lead in event.ecg.leads)

    # Required fields
    assert event.event_id is not None
    assert event.uuid is not None
    assert event.timestamp > 0
```

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| TransCovNet accuracy (69%) on simulated data | Misclassification in production | Fine-tune on real clinical data; use rule engine as fallback |
| Untrained heatmap decoder | Cannot use neural beat detection | Signal-based detector is default; heatmap opt-in when trained |
| Signal-based beat detector on noisy data | Missed beats or false positives | Quality assessment gates noisy signals; adaptive thresholds |
| HDF5 file format variations | Data loading failures | Strict schema validation in loader; comprehensive error messages |
| LLM API latency for queries | Slow query responses | Query router sends Type A (deterministic) queries to local handler; only Type B uses LLM |
| Model weights not available in early phases | Can't run full pipeline | Each CLI phase can run with random/mock weights for shape validation |

## Migration Plan
Not applicable — greenfield project. No existing system to migrate from.

## Open Questions
1. ~~Should PPG and RESP auxiliary signals be used in any pipeline stage beyond data loading?~~ Currently loaded but not used in interpretation.
2. ~~What specific clinical rules should be included in the initial rule engine release?~~ Defined in `config/rules_config.yaml`.
3. ~~Which LLM provider (Anthropic vs OpenAI) should be the default?~~ Configurable via `Settings.llm_provider`, default Anthropic.
4. ~~Should the system support batch processing of multiple events in a single API call?~~ Yes, via `run_inference.py --all`.
5. When should the heatmap decoder be trained, and what training data is needed?
6. What real clinical data will be used for TransCovNet fine-tuning to improve beyond 69% accuracy?
