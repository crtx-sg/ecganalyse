# Plan: Phase 1 — Signal Preprocessing

## Context

Phase 0 (data loading) is complete. Phase 1 adds signal quality assessment, a U-Net denoiser, and preprocessing utilities. The `QualityReport` dataclass, `PreprocessingConfig`, and `SignalQualityError` already exist as stubs. The `src/preprocessing/` and `tests/test_preprocessing/` directories exist with empty `__init__.py` files.

User guidance: keep it simple, CLI-testable, good test coverage.

## Files to create

| # | File | Purpose |
|---|------|---------|
| 1 | `src/preprocessing/quality.py` | `SignalQualityAssessor` — SQI per lead, lead-off, saturation, pacer, overall |
| 2 | `src/preprocessing/denoiser.py` | `ECGDenoiser` — U-Net, residual learning, `[B,7,2400] → [B,7,2400]` |
| 3 | `src/preprocessing/utils.py` | `bandpass_filter`, `remove_baseline_wander`, `normalize_leads` |
| 4 | `tests/test_preprocessing/test_quality.py` | Unit tests for quality assessor |
| 5 | `tests/test_preprocessing/test_denoiser.py` | Unit tests for denoiser (shape, residual, energy) |
| 6 | `tests/test_preprocessing/test_utils.py` | Unit tests for filter/normalize utilities |
| 7 | `tests/gates/test_phase1_gate.py` | Gate tests: QualityReport + denoised tensor contracts |
| 8 | `scripts/cli_phase1.py` | CLI harness: HDF5 → quality + denoise → JSON report |
| 9 | `tests/test_integration/test_phase1_integration.py` | Integration: AlarmEvent → QualityReport + denoised |

## Files to modify

None. All infrastructure (`schemas.py`, `exceptions.py`, `settings.py`, `model_config.yaml`) already has the needed stubs.

---

## Implementation details

### 1. `src/preprocessing/quality.py` — SignalQualityAssessor

```python
class SignalQualityAssessor:
    def __init__(self, config: PreprocessingConfig | None = None)
    def assess(self, ecg: ECGData) -> QualityReport
    def compute_sqi(self, signal: np.ndarray, fs: int) -> float
    def detect_lead_off(self, signal: np.ndarray) -> bool
    def detect_saturation(self, signal: np.ndarray) -> bool
    def check_pacer_presence(self, extras: dict) -> bool
```

SQI computation (per-lead, 0.0–1.0):
- Variance check (low variance → lead-off → SQI 0)
- Kurtosis (physiological ECG kurtosis ~3–10; flat/noise outside range penalised)
- High-frequency power ratio (>40Hz noise vs total power)
- Baseline stability (std of low-pass filtered signal)
- Combine sub-metrics as weighted average

Overall SQI = mean of per-lead SQI for usable leads.

Quality flags: `lead_off_{lead}`, `saturation_{lead}`, `pacer_detected`, `signal_unusable`, `high_noise`.

Noise level classification: SQI >= 0.7 → "low", >= 0.4 → "moderate", else "high".
Baseline stability: based on baseline wander amplitude.

### 2. `src/preprocessing/denoiser.py` — ECGDenoiser

Simple U-Net with residual learning (output = input - predicted_noise).

```python
class ECGDenoiser(nn.Module):
    def __init__(self, in_channels=7, base_channels=32, depth=4)
    def forward(self, x: Tensor) -> Tensor  # [B,7,2400] → [B,7,2400]
```

Architecture (from `model_config.yaml`):
- Encoder: `depth` blocks of Conv1d → BatchNorm → ReLU → MaxPool1d(2)
- Bottleneck: Conv1d → BatchNorm → ReLU
- Decoder: `depth` blocks of ConvTranspose1d → concat skip → Conv1d → BatchNorm → ReLU
- Final: Conv1d → 7 channels (predicted noise)
- Output: `x - noise` (residual)

No trained weights needed — works with random weights for shape validation. The training script (task 2.9) is deferred per user guidance ("simple, cli, testing").

### 3. `src/preprocessing/utils.py`

```python
def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 4) -> np.ndarray
def remove_baseline_wander(signal: np.ndarray, fs: float, cutoff: float = 0.5) -> np.ndarray
def normalize_leads(ecg_array: np.ndarray) -> np.ndarray  # [7,2400] → [7,2400], zero-mean unit-var per lead
```

Uses `scipy.signal.butter` + `sosfiltfilt` for zero-phase filtering.

### 4–6. Unit tests

**test_quality.py:**
- Clean signal → overall_sqi >= 0.85, all leads usable, noise_level "low"
- Flat lead (lead-off) → that lead excluded, SQI < 0.3, flag `lead_off_{lead}`
- Pacer extras → `pacer_detected` flag
- All-noise signal → `signal_unusable` flag

**test_denoiser.py:**
- Forward pass shape: `[1,7,2400]` → `[1,7,2400]`, float32
- Residual: clean input → output ≈ input (noise prediction near zero — not exact with random weights, just test the subtraction structure)
- Energy preservation: output RMS within 50% of input RMS (with random weights)
- Batch dimension: `[4,7,2400]` works

**test_utils.py:**
- Bandpass attenuates 60Hz (add 60Hz sine, verify attenuation > 20dB)
- Bandpass preserves QRS (signal energy ratio before/after)
- Normalize: output mean ≈ 0, std ≈ 1 per lead
- Edge case: constant signal normalize (std=0 → no division error)

### 7. Gate tests — `tests/gates/test_phase1_gate.py`

Contract validation:
- QualityReport: overall_sqi in [0,1], lead_sqi has 7 keys in [0,1], usable_leads ⊆ all leads, noise_level in {"low","moderate","high"}, baseline_stability in {"stable","moderate","unstable"}
- Denoised tensor: shape [7,2400], dtype float32, values in [-10,10]

### 8. CLI harness — `scripts/cli_phase1.py`

```
python scripts/cli_phase1.py <hdf5_file> <event_id>
```

Output JSON:
```json
{
  "event_id": "event_1001",
  "quality_report": {
    "overall_sqi": 0.87,
    "lead_sqi": {"ECG1": 0.91, ...},
    "usable_leads": [...],
    "excluded_leads": [],
    "quality_flags": [],
    "noise_level": "low",
    "baseline_stability": "stable"
  },
  "denoising_summary": {
    "ECG1": {"rms_before": 0.34, "rms_after": 0.31},
    ...
  }
}
```

### 9. Integration test

Loads HDF5 via Phase 0 loader → runs SignalQualityAssessor → runs ECGDenoiser → validates output types and shapes end-to-end.

---

## Verification

```bash
# Unit tests
pytest tests/test_preprocessing/ -v

# Gate tests
pytest tests/gates/test_phase1_gate.py -v

# Integration test
pytest tests/test_integration/test_phase1_integration.py -v

# CLI harness
python scripts/cli_phase1.py data/samples/PT1234_2024-01.h5 event_1001

# All tests together
pytest tests/test_preprocessing/ tests/gates/test_phase1_gate.py tests/test_integration/test_phase1_integration.py -v
```
