## ADDED Requirements

### Requirement: Signal Quality Assessment
The system SHALL compute a Signal Quality Index (SQI) for each of the 7 ECG leads and an overall composite SQI. The assessor SHALL detect lead-off conditions (flat line or rail voltage), signal saturation, excessive noise, baseline instability, and pacemaker artifacts (via ECGData extras field). The output SHALL be a `QualityReport` containing per-lead SQI scores (0.0–1.0), lists of usable and excluded leads, quality flags, noise level classification, and baseline stability classification.

#### Scenario: Assess clean 7-lead ECG
- **WHEN** a 7-lead ECG with minimal noise and no artifacts is assessed
- **THEN** `QualityReport.overall_sqi` SHALL be >= 0.85
- **AND** all 7 leads SHALL appear in `usable_leads`
- **AND** `excluded_leads` SHALL be empty
- **AND** `noise_level` SHALL be `"low"`

#### Scenario: Detect lead-off on one lead
- **WHEN** lead aVR contains a flat-line signal (near-zero variance)
- **THEN** `lead_sqi["aVR"]` SHALL be < 0.3
- **AND** `"aVR"` SHALL appear in `excluded_leads`
- **AND** `quality_flags` SHALL contain `"lead_off_aVR"`

#### Scenario: Detect pacemaker presence
- **WHEN** ECGData.extras contains `{"pacer_info": 1}`
- **THEN** `check_pacer_presence()` SHALL return `True`
- **AND** `quality_flags` SHALL contain `"pacer_detected"`

#### Scenario: Handle fully noisy signal
- **WHEN** all leads have SQI below 0.3
- **THEN** `QualityReport.overall_sqi` SHALL be < 0.3
- **AND** `quality_flags` SHALL contain `"signal_unusable"`

### Requirement: Learned ECG Denoiser
The system SHALL provide a U-Net-based neural denoiser that accepts 7-lead ECG input of shape `[batch, 7, 2400]` and outputs denoised signals of the same shape. The denoiser SHALL use residual learning (output = input - predicted_noise). The denoiser SHALL preserve pacemaker spikes when present. The model SHALL be trainable via a provided training script.

#### Scenario: Denoise 7-lead ECG
- **WHEN** a noisy 7-lead ECG tensor of shape `[1, 7, 2400]` is passed through the denoiser
- **THEN** the output SHALL have shape `[1, 7, 2400]`
- **AND** the output dtype SHALL match the input dtype (float32)

#### Scenario: Residual learning structure
- **WHEN** the denoiser is applied to a clean signal (no noise)
- **THEN** the predicted noise SHALL be near-zero
- **AND** the output SHALL be approximately equal to the input (within tolerance)

#### Scenario: Preserve signal energy
- **WHEN** the denoiser is applied
- **THEN** the output signal energy (RMS) SHALL be within 50% of the input signal energy (signals are not zeroed out)

### Requirement: Preprocessing Utilities
The system SHALL provide utility functions for bandpass filtering (0.5–40Hz for ECG), baseline wander removal, and per-lead normalization (zero-mean, unit-variance). These utilities SHALL operate on numpy arrays and be usable independently of the neural denoiser.

#### Scenario: Bandpass filter ECG signal
- **WHEN** a raw ECG signal with 60Hz powerline interference is filtered with bandpass 0.5–40Hz
- **THEN** the 60Hz component SHALL be attenuated by at least 20dB
- **AND** the QRS complex morphology SHALL be preserved

#### Scenario: Normalize per-lead
- **WHEN** a 7-lead ECG array of shape `[7, 2400]` is normalized
- **THEN** each lead SHALL have mean approximately 0.0 and std approximately 1.0

### Requirement: Phase 1 CLI Harness
The system SHALL provide a CLI script `scripts/cli_phase1.py` that loads an HDF5 event, runs quality assessment and denoising, and prints a JSON report to stdout containing the QualityReport and signal statistics (before/after denoising).

#### Scenario: CLI runs Phase 1 pipeline
- **WHEN** `python -m scripts.cli_phase1 data/samples/test.h5 event_1001` is run
- **THEN** stdout SHALL contain a JSON object with `quality_report` (overall_sqi, lead_sqi, flags) and `denoising_summary` (per-lead RMS before/after)
