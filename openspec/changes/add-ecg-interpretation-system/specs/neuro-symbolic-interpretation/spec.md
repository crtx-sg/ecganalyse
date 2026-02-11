## ADDED Requirements

### Requirement: Symbolic Calculation Engine
The system SHALL compute clinical ECG measurements from fiducial points using deterministic formulas. All calculations SHALL produce auditable calculation traces as human-readable strings showing the formula, input values, and result. Measurements SHALL include: PR interval (ms), QRS duration (ms), QT interval (ms), QTc Bazett (ms), QTc Fridericia (ms), RR intervals (mean, std, min, max in ms), and heart rate (bpm). The engine SHALL compute both per-beat and global (median/mean across beats) measurements.

#### Scenario: Compute PR interval with trace
- **WHEN** a beat has P-onset at sample 156 and QRS-onset at sample 192 at 200Hz
- **THEN** PR interval SHALL be `(192 - 156) / 200 * 1000 = 180 ms`
- **AND** the calculation trace SHALL be `"(sample_192 - sample_156) / 200 * 1000 = 180 ms"`

#### Scenario: Compute QTc Bazett
- **WHEN** QT interval is 380ms and RR interval is 832ms
- **THEN** QTc Bazett SHALL be `380 / sqrt(832/1000) = 416.5 ms`
- **AND** the calculation trace SHALL show the formula with actual values

#### Scenario: Compute heart rate from RR intervals
- **WHEN** RR intervals across beats have mean of 832ms
- **THEN** heart rate SHALL be `60000 / 832 = 72.1 bpm`

#### Scenario: Handle missing fiducials
- **WHEN** a beat is missing P-wave fiducials
- **THEN** PR interval SHALL be `None` for that beat
- **AND** global PR interval SHALL be computed from beats that have P-wave data
- **AND** the trace SHALL note `"PR interval: computed from N of M beats (P-wave absent in M-N beats)"`

### Requirement: Rule-Based Reasoning Engine
The system SHALL apply clinical rules to measurements and beat morphology to generate findings. Rules SHALL be defined in `config/rules_config.yaml` and loaded at startup. The engine SHALL classify rhythm (normal sinus, sinus bradycardia, sinus tachycardia, atrial fibrillation, etc.), detect conduction abnormalities (AV block, bundle branch block), identify morphology findings (ST elevation/depression, T-wave inversion, axis deviation), and flag ischemia markers. Each finding SHALL have a severity level (`normal`, `mild`, `moderate`, `severe`, `critical`) and confidence score.

#### Scenario: Classify normal sinus rhythm
- **WHEN** heart rate is 60–100 bpm, RR intervals are regular (std < 120ms), P-waves are present before each QRS, and PR interval is 120–200ms
- **THEN** rhythm classification SHALL be `"normal_sinus_rhythm"` with confidence >= 0.90
- **AND** a finding with `category="rhythm"` and `severity="normal"` SHALL be generated

#### Scenario: Detect sinus tachycardia
- **WHEN** heart rate is 105 bpm with regular rhythm and 1:1 P:QRS relationship
- **THEN** rhythm classification SHALL be `"sinus_tachycardia"`
- **AND** finding severity SHALL be `"mild"`

#### Scenario: Detect prolonged QTc
- **WHEN** QTc Bazett is 480ms (male) or 490ms (female)
- **THEN** a finding SHALL be generated with `finding="prolonged_qtc"`, `category="conduction"`, `severity="moderate"`

#### Scenario: Multiple findings
- **WHEN** analysis reveals both sinus tachycardia and prolonged QTc
- **THEN** both findings SHALL be present in the findings list
- **AND** each SHALL have its own finding_id, confidence, and evidence

### Requirement: Vitals Context Integration
The system SHALL integrate contextual vital signs from the alarm event into the interpretation. The integrator SHALL: (1) validate ECG-derived heart rate against monitor HR and report discrepancy, (2) check all vitals against their configured thresholds and generate findings for violations, (3) generate cross-modal clinical findings (e.g., tachycardia with hypertension suggesting stress response). Vitals-based findings SHALL use `category="vital"`.

#### Scenario: HR validation — consistent
- **WHEN** ECG-derived HR is 72 bpm and monitor HR is 73 bpm
- **THEN** a finding SHALL be generated: `finding="hr_ecg_monitor_consistent"`, `severity="normal"`, with evidence showing `ecg_hr=72, monitor_hr=73, difference=1`

#### Scenario: HR validation — discrepant
- **WHEN** ECG-derived HR is 72 bpm and monitor HR is 95 bpm
- **THEN** a finding SHALL be generated: `finding="hr_ecg_monitor_discrepant"`, `severity="moderate"`, with evidence showing `difference=23`

#### Scenario: SpO2 below threshold
- **WHEN** SpO2 value is 85% and lower_threshold is 90%
- **THEN** a finding SHALL be generated: `finding="spo2_below_threshold"`, `category="vital"`, `severity="severe"`

#### Scenario: All vitals normal
- **WHEN** all vitals are within their threshold ranges
- **THEN** a finding SHALL be generated: `finding="vitals_within_normal"`, `category="vital"`, `severity="normal"`

### Requirement: JSON Feature Assembly
The system SHALL assemble a complete JSON Feature Assembly document (schema version 1.1) containing: `event_context` (event_id, uuid, timestamp, patient_id, alarm offset, device info), `metadata` (strip info, leads, pacer status), `vitals_context` (all vitals with thresholds and violation flags), `quality` (SQI scores, usable leads, flags), `global_measurements` (HR, RR, PR, QRS, QT, QTc with HR validation), `rhythm` (classification, regularity, P-wave analysis), `beats` (per-beat fiducials and intervals), `findings` (all clinical and vital findings), `calculation_traces` (all auditable traces), and `summary` (primary interpretation, categories, critical findings). The assembly SHALL include `schema_version`, `generated_at`, and `processing_time_ms`.

#### Scenario: Assemble complete JSON for normal ECG
- **WHEN** a normal sinus rhythm ECG with all vitals normal is assembled
- **THEN** the output SHALL be valid JSON with all required top-level keys
- **AND** `schema_version` SHALL be `"1.1"`
- **AND** `findings` SHALL include rhythm, validation, and vital findings
- **AND** `calculation_traces` SHALL contain at least traces for `pr_interval`, `heart_rate`, and `qtc_bazett`
- **AND** `summary.abnormality_count` SHALL be `0`

#### Scenario: Assembly preserves event context
- **WHEN** AlarmEvent has `event_id="event_1001"`, `uuid="550e..."`, `timestamp=1704537600.0`
- **THEN** `event_context.event_id` SHALL be `"event_1001"`
- **AND** `event_context.event_uuid` SHALL be `"550e..."`
- **AND** `event_context.event_timestamp` SHALL be `1704537600.0`

### Requirement: Phase 4 CLI Harness
The system SHALL provide a CLI script `scripts/cli_phase4.py` that runs the full pipeline through interpretation and prints the complete JSON Feature Assembly to stdout.

#### Scenario: CLI runs Phase 4 pipeline
- **WHEN** `python -m scripts.cli_phase4 data/samples/test.h5 event_1001` is run
- **THEN** stdout SHALL contain the complete JSON Feature Assembly
- **AND** the JSON SHALL be valid and parseable
- **AND** `calculation_traces` SHALL be populated
