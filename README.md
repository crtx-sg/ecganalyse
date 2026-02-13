# ECG Interpretation System

AI-powered ECG interpretation system that analyzes 7-lead electrocardiogram signals from clinical alarm events stored in HDF5 format. Combines a neural condition classifier (ECG-TransCovNet) with classical signal-based beat detection and symbolic rule-based interpretation to produce auditable clinical findings.

## Architecture

The system uses a **neuro-symbolic pipeline** — neural networks handle pattern recognition (arrhythmia classification), while deterministic symbolic engines compute clinical measurements with auditable traces (IEC 62304).

```
HDF5 File
    |
    v
Phase 0: Data Loading          HDF5AlarmEventLoader -> AlarmEvent
    |
    v
Phase 1: Signal Quality        SignalQualityAssessor -> QualityReport
    |
    v
Phase 3a: Beat Detection       SignalBasedBeatDetector (classical DSP) -> List[Beat]
    |
Phase 3b: Condition Classify   ConditionClassifier (ECG-TransCovNet, 16 classes) -> ConditionPrediction
    |
    v
Phase 4: Interpretation        SymbolicCalculationEngine + RuleBasedReasoningEngine
    |                          + VitalsContextIntegrator -> JSON Feature Assembly
    v
Output: JSON report with measurements, rhythm, findings, traces
```

### Key Components

| Component | Description |
|-----------|-------------|
| **ECG-TransCovNet** | 16-class cardiac condition classifier (CNN backbone + Transformer encoder-decoder) |
| **SignalBasedBeatDetector** | Classical DSP beat detection: bandpass filter + adaptive R-peak finding + fiducial estimation |
| **SymbolicCalculationEngine** | Deterministic interval/HR computation with calculation traces |
| **RuleBasedReasoningEngine** | YAML-driven clinical rules for rhythm classification and findings |
| **VitalsContextIntegrator** | Cross-modal validation (ECG HR vs monitor HR, threshold checks) |
| **JSONAssembler** | Assembles complete JSON Feature Assembly v1.1 |

### Supported Conditions (16 classes)

Normal Sinus, Sinus Bradycardia, Sinus Tachycardia, Atrial Fibrillation, Atrial Flutter, PAC, SVT, PVC, Ventricular Tachycardia, Ventricular Fibrillation, LBBB, RBBB, AV Block 1st degree, AV Block 2nd degree Type 1, AV Block 2nd degree Type 2, ST Elevation

## Quick Start

### Prerequisites

- Python 3.11+
- Trained TransCovNet weights at `models/ecg_transcovnet/best_model.pt`

### Install

```bash
pip install -e ".[dev]"
```

### Run Inference

```bash
# Single event — full pipeline (classification + beats + measurements)
python scripts/run_inference.py data/test_conditions/normal_sinus_75_medium.h5 event_1001

# Classification only (skip beat detection)
python scripts/run_inference.py data/test_conditions/afib_120_medium.h5 event_1001 --no-beats

# Skip HR or interval measurements separately
python scripts/run_inference.py data/test_conditions/afib_120_medium.h5 event_1001 --no-hr
python scripts/run_inference.py data/test_conditions/afib_120_medium.h5 event_1001 --no-intervals

# All events in a file
python scripts/run_inference.py data/samples/PT9401_2026-02.h5 --all

# Write output to file
python scripts/run_inference.py data/test_conditions/afib_120_medium.h5 event_1001 -o result.json
```

### Batch Test Across Conditions

```bash
# Test all 13 conditions and print summary table
python scripts/test_pipeline.py

# Classification-only mode
python scripts/test_pipeline.py --no-beats
```

### Visual Report

```bash
# Generate PNG clinical report
python scripts/visualize_phase4.py data/test_conditions/afib_120_medium.h5 event_1001

# All leads + JSON dump
python scripts/visualize_phase4.py data/test_conditions/afib_120_medium.h5 event_1001 --all-leads --json
```

### Phase-by-Phase CLI

```bash
python scripts/cli_phase0.py data/samples/PT1234_2024-01.h5 event_1001   # Data loading
python scripts/cli_phase1.py data/samples/PT1234_2024-01.h5 event_1001   # Quality + denoising
python scripts/cli_phase2.py data/samples/PT1234_2024-01.h5 event_1001   # Feature encoding
python scripts/cli_phase3.py data/samples/PT1234_2024-01.h5 event_1001   # Beat detection
python scripts/cli_phase4.py data/samples/PT1234_2024-01.h5 event_1001   # Full interpretation
```

## Project Structure

```
ecganalyse/
├── config/
│   ├── settings.py              # LoaderConfig, PreprocessingConfig, PipelineConfig
│   ├── model_config.yaml        # Model hyperparameters
│   └── rules_config.yaml        # Clinical rules definitions
├── src/
│   ├── data/                    # Phase 0: HDF5 loading, event/vitals parsing, dataset
│   ├── preprocessing/           # Phase 1: Signal quality, denoiser, utilities
│   ├── encoding/                # Phase 2: TransCovNet, Mamba, Swin, fusion
│   ├── prediction/              # Phase 3: Beat detection, heatmap decoder, fiducials, condition classifier
│   ├── interpretation/          # Phase 4: Symbolic engine, rules, vitals context, JSON assembly
│   ├── query/                   # Phase 5: Query router, deterministic/LLM handlers
│   ├── api/                     # Phase 6: FastAPI service
│   ├── simulator/               # ECG signal simulator (16 conditions)
│   ├── training/                # Training utilities, datasets, losses, metrics
│   └── ecg_system/              # Core schemas, exceptions
├── scripts/
│   ├── run_inference.py         # Unified inference entry point
│   ├── test_pipeline.py         # Batch test across conditions
│   ├── cli_phase{0-4}.py        # Per-phase CLI harnesses
│   ├── visualize_phase4.py      # Visual clinical report generator
│   ├── generate_*.py            # Test data generation scripts
│   └── train.py                 # Training script
├── models/
│   ├── ecg_transcovnet/         # Trained TransCovNet checkpoint
│   └── weights/                 # Stage A denoiser weights
├── data/
│   ├── samples/                 # Sample HDF5 files
│   └── test_conditions/         # 13 test condition files
├── tests/                       # pytest suite (351 tests)
└── openspec/                    # Specification-driven development docs
```

## Pipeline Configuration

The `PipelineConfig` dataclass controls which pipeline stages run:

```python
from config.settings import PipelineConfig

config = PipelineConfig(
    enable_beat_analysis=True,         # Beat detection + fiducials
    enable_heart_rate=True,            # HR from R-peak intervals
    enable_interval_measurements=True, # PR, QRS, QT, QTc
    beat_detector="signal",            # "signal" (DSP) or "heatmap" (neural)
)
```

The `--heatmap` flag on `cli_phase4.py` and `visualize_phase4.py` opts into the neural heatmap decoder pipeline (requires trained heatmap model weights, which are not yet available).

## Testing

```bash
# Full test suite (351 tests)
python -m pytest tests/ --ignore=tests/test_integration -k "not gate" -q

# Specific module
python -m pytest tests/test_interpretation/ -q

# Integration tests
python -m pytest tests/test_integration/ -q
```

## Training

```bash
# Train ECG-TransCovNet condition classifier
python train_ecg_transcovnet.py

# Evaluate model
python scripts/evaluate.py
```

## Data Format

Input: HDF5 alarm event files (`PatientID_YYYY-MM.h5`)

```
PatientID_YYYY-MM.h5
├── metadata/
├── event_1001/
│   ├── ecg/          # 7 leads x 2400 samples (200Hz, 12s)
│   ├── ppg/          # 900 samples (75Hz)
│   ├── resp/         # 400 samples (33.33Hz)
│   ├── vitals/       # HR, SpO2, BP, RespRate, Temp
│   ├── timestamp
│   └── uuid
├── event_1002/
└── ...
```

Generate test data:

```bash
python scripts/generate_conditions_test_set.py    # 13 cardiac conditions
python scripts/generate_sample_hdf5.py            # Sample patient files
```
