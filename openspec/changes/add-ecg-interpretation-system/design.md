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

### Non-Goals
- Real-time streaming ECG processing (batch/event-based only)
- 12-lead ECG support (7-lead only for this release)
- Training pipeline optimization (basic training scripts only)
- Mobile or embedded deployment
- Multi-patient concurrent processing optimization
- FDA/CE regulatory submission (documentation support only)

## Architecture Decisions

### Decision 1: Six-Stage Pipeline Architecture
**What**: Linear pipeline with well-defined interfaces between stages: Data Loading → Preprocessing → Encoding → Prediction → Interpretation → Query Interface.
**Why**: Each stage is independently testable, replaceable, and maps cleanly to the regulatory V-model. Stages communicate through typed dataclasses, enabling contract-based testing (gate tests) at every boundary.
**Alternatives considered**:
- Monolithic model (end-to-end neural): Rejected — no explainability, no calculation traces
- Microservices per stage: Over-engineered for current scale; can be split later

### Decision 2: Neuro-Symbolic Dual Architecture
**What**: Neural networks (Mamba + Swin) for feature extraction and fiducial detection; symbolic engine for measurement calculation and clinical rule application.
**Why**: Neural components handle pattern recognition in noisy signals; symbolic components provide deterministic, auditable measurements required by IEC 62304. This separation means the LLM never sees raw signals — only structured features.
**Alternatives considered**:
- Pure neural interpretation: No auditability; fails regulatory requirements
- Pure symbolic/rule-based: Insufficient for complex morphology detection in noisy real-world signals

### Decision 3: Dual-Path Feature Encoding (Mamba + Swin)
**What**: ECG-Mamba captures long-range temporal dependencies (rhythm, RR intervals); 1D-Swin Transformer captures local morphological features (wave shapes, intervals). Features are fused before prediction.
**Why**: ECG analysis requires both global context (is the rhythm regular?) and local precision (where exactly does the QRS start?). Single-encoder approaches compromise one for the other.
**Alternatives considered**:
- Single Transformer: Poor long-range efficiency for 2400-sample sequences
- Single Mamba: Less precise local feature extraction
- CNN-only: Limited receptive field without excessive depth

### Decision 4: Graph Attention Network for Multi-Lead Fusion
**What**: 7-node graph where each node is a lead, with anatomically-informed adjacency (e.g., ECG1↔ECG2, vVX connected to limb cluster). GAT learns cross-lead attention.
**Why**: Different leads provide different views of the same cardiac event. Anatomically-informed graph structure encodes clinical knowledge about lead relationships. Attention weights provide explainability.
**Alternatives considered**:
- Simple concatenation: Loses spatial relationships between leads
- 2D CNN on lead matrix: Imposes arbitrary spatial ordering
- Full attention across leads: Works but doesn't encode anatomical priors

### Decision 5: Feature-Augmented LLM (Not Signal-Augmented)
**What**: The LLM receives only structured JSON (measurements, findings, vitals, traces) and clinical queries. Never raw signals, embeddings, or neural features.
**Why**: Regulatory requirement — every LLM response must be traceable to deterministic measurements. Also simpler to validate, test, and audit. LLM serves as a clinical reasoning and natural language interface, not a signal processor.
**Alternatives considered**:
- Signal tokens to LLM: No auditability; hallucination risk on measurements
- No LLM (rule-based NLG only): Limited expressiveness for complex clinical queries

### Decision 6: Per-Phase CLI Test Harness
**What**: Each pipeline phase has a standalone CLI script (`scripts/cli_phaseN.py`) that accepts HDF5 input, runs that phase (with mock/pretrained predecessors as needed), and outputs structured JSON to stdout.
**Why**: Enables manual verification by clinical engineers, regression testing in CI, and phase-by-phase development without waiting for downstream stages.
**Alternatives considered**:
- API-only testing: Requires full stack; too heavyweight for phase development
- Notebook-only testing: Not automatable in CI

### Decision 7: Gate Tests at Phase Boundaries
**What**: Pytest-based gate tests that validate the output schema and value ranges of each pipeline phase against its contract (input/output dataclass shapes, required fields, value bounds).
**Why**: Catches interface drift between phases early. If Phase 2's output shape changes, Phase 3's gate test fails immediately — before any logic errors cascade. Critical for multi-developer parallel work.
**Alternatives considered**:
- Integration tests only: Catch issues too late
- Manual contract review: Error-prone, doesn't scale

## Data Flow

```
HDF5 File
    │
    ▼
┌─────────────────────────────────┐
│ Phase 0: Data Loading           │
│  Input:  HDF5 filepath + event  │
│  Output: AlarmEvent dataclass   │
│  CLI:    cli_phase0             │
│  Tests:  test_data/             │
└──────────────┬──────────────────┘
               │ AlarmEvent
               ▼
┌─────────────────────────────────┐
│ Phase 1: Signal Preprocessing   │
│  Input:  AlarmEvent.ecg (7,2400)│
│  Output: QualityReport +        │
│          denoised ECG (7,2400)  │
│  CLI:    cli_phase1             │
│  Tests:  test_preprocessing/    │
└──────────────┬──────────────────┘
               │ (QualityReport, Tensor[7,2400])
               ▼
┌─────────────────────────────────┐
│ Phase 2: Feature Encoding       │
│  Input:  denoised ECG (7,2400)  │
│  Output: fused features         │
│          (7, seq_len, d_model)  │
│  CLI:    cli_phase2             │
│  Tests:  test_encoding/         │
└──────────────┬──────────────────┘
               │ Tensor[7, seq_len, d_model]
               ▼
┌─────────────────────────────────┐
│ Phase 3: Dense Prediction       │
│  Input:  fused features         │
│  Output: heatmaps + fiducial    │
│          points per lead/beat   │
│  CLI:    cli_phase3             │
│  Tests:  test_prediction/       │
└──────────────┬──────────────────┘
               │ List[Beat] with fiducials
               ▼
┌─────────────────────────────────┐
│ Phase 4: Interpretation         │
│  Input:  fiducials + AlarmEvent │
│  Output: JSON Feature Assembly  │
│          (measurements, rhythm, │
│           findings, vitals,     │
│           traces)               │
│  CLI:    cli_phase4             │
│  Tests:  test_interpretation/   │
└──────────────┬──────────────────┘
               │ Dict (JSON Assembly)
               ▼
┌─────────────────────────────────┐
│ Phase 5: Clinical Query         │
│  Input:  JSON Assembly + query  │
│  Output: natural language answer│
│  CLI:    cli_phase5             │
│  Tests:  test_query/            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Phase 6: API Service            │
│  FastAPI wrapping full pipeline │
│  CLI:    uvicorn                │
│  Tests:  test_api/              │
└─────────────────────────────────┘
```

## Testing Strategy Per Phase

Each phase includes four test categories:

| Category | Purpose | Tool | When |
|----------|---------|------|------|
| **Static** | Type checking, linting | mypy, ruff | Pre-commit, CI |
| **Unit** | Individual function/class correctness | pytest | Every commit |
| **Gate** | Input/output contract validation at phase boundaries | pytest | Every commit |
| **Integration** | End-to-end within phase, cross-phase | pytest | PR merge, nightly |

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
| Mamba-SSM library instability | Encoding phase blocked | Fallback to pure Transformer; `foundation.py` adapter pattern |
| torch-geometric installation complexity | Dev environment setup friction | Docker dev container with pre-built wheels |
| HDF5 file format variations | Data loading failures | Strict schema validation in loader; comprehensive error messages |
| LLM API latency for queries | Slow query responses | Query router sends Type A (deterministic) queries to local handler; only Type B uses LLM |
| Model weights not available in early phases | Can't run full pipeline | Each CLI phase can run with random/mock weights for shape validation |

## Migration Plan
Not applicable — greenfield project. No existing system to migrate from.

## Open Questions
1. Should PPG and RESP auxiliary signals be used in any pipeline stage beyond data loading?
2. What specific clinical rules should be included in the initial rule engine release?
3. Which LLM provider (Anthropic vs OpenAI) should be the default?
4. Should the system support batch processing of multiple events in a single API call?
