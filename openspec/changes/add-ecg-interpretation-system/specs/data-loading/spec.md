## ADDED Requirements

### Requirement: HDF5 File Loading
The system SHALL load and parse HDF5 clinical alarm event files conforming to the `PatientID_YYYY-MM.h5` format. The system SHALL support opening files as context managers and SHALL list all event IDs within a file. The system SHALL extract global metadata including patient_id, sampling rates (ECG: 200Hz, PPG: 75Hz, RESP: 33.33Hz), alarm timing (offset, before/after seconds), data quality score, device info, and max vital history.

#### Scenario: Load valid HDF5 file and list events
- **WHEN** a valid HDF5 file `PT1234_2024-01.h5` containing events `event_1001`, `event_1002` is provided
- **THEN** the loader SHALL open the file without error
- **AND** `list_events()` SHALL return `["event_1001", "event_1002"]`
- **AND** `load_metadata()` SHALL return a `FileMetadata` with `patient_id="PT1234"` and `sampling_rate_ecg=200.0`

#### Scenario: Handle missing or corrupt HDF5 file
- **WHEN** a non-existent or corrupt file path is provided
- **THEN** the loader SHALL raise `HDF5LoadError` with a descriptive message

#### Scenario: Handle HDF5 file with no events
- **WHEN** a valid HDF5 file with only metadata and no event groups is provided
- **THEN** `list_events()` SHALL return an empty list

### Requirement: Alarm Event ECG Parsing
The system SHALL extract 7-lead ECG data from each alarm event group. The leads SHALL be ECG1, ECG2, ECG3, aVR, aVL, aVF, and vVX. Each lead SHALL contain 2400 float32 samples (12 seconds at 200Hz, gzip-compressed in HDF5). The parser SHALL also extract extras (pacer_info, pacer_offset, etc.) from the ECG group's extras dataset as JSON.

#### Scenario: Parse 7-lead ECG from alarm event
- **WHEN** event `event_1001` is parsed from a valid HDF5 file
- **THEN** the result SHALL contain an `ECGData` with `signals` dict having exactly 7 keys: `ECG1`, `ECG2`, `ECG3`, `aVR`, `aVL`, `aVF`, `vVX`
- **AND** each signal array SHALL have shape `(2400,)` and dtype `float32`
- **AND** `sample_rate` SHALL be `200`
- **AND** `duration_sec` SHALL be `12.0`

#### Scenario: Parse ECG extras with pacer info
- **WHEN** an event's ECG extras contain `{"pacer_info": 1, "pacer_offset": 5}`
- **THEN** `ECGData.extras` SHALL contain `{"pacer_info": 1, "pacer_offset": 5}`

#### Scenario: Handle missing ECG lead gracefully
- **WHEN** an event is missing one or more expected leads
- **THEN** the parser SHALL raise `EventParseError` identifying the missing leads

### Requirement: Auxiliary Signal Parsing
The system SHALL extract PPG signals (75Hz, 900 samples) and respiratory signals (33.33Hz, ~400 samples) from each alarm event when present. Missing auxiliary signals SHALL NOT cause parsing failure.

#### Scenario: Parse PPG and RESP signals
- **WHEN** event `event_1001` contains both `ppg/PPG` and `resp/RESP` datasets
- **THEN** PPG SHALL be returned as `PPGData` with shape `(900,)` and sample_rate `75.0`
- **AND** RESP SHALL be returned as `RespData` with shape `(400,)` and sample_rate `33.33`

#### Scenario: Event with missing auxiliary signals
- **WHEN** an event has no `ppg/` or `resp/` group
- **THEN** `AlarmEvent.ppg` and `AlarmEvent.resp` SHALL be `None`
- **AND** ECG parsing SHALL proceed normally

### Requirement: Vitals Data Parsing
The system SHALL extract vital sign measurements from each alarm event. Supported vital types are: HR (bpm), Pulse (bpm), SpO2 (%), Systolic (mmHg), Diastolic (mmHg), RespRate (brpm), Temp (°C), and XL_Posture (degrees). Each vital SHALL include value, units, timestamp, and extras (containing upper/lower thresholds). The system SHALL detect threshold violations by comparing values against their thresholds.

#### Scenario: Parse all vital types
- **WHEN** an event's `vitals/` group contains HR, SpO2, Systolic, Diastolic, RespRate, Temp, and XL_Posture subgroups
- **THEN** `VitalsData` SHALL contain a `VitalMeasurement` for each present vital
- **AND** each measurement SHALL have `name`, `value`, `units`, and `timestamp`
- **AND** thresholds SHALL be accessible via `upper_threshold` and `lower_threshold` properties

#### Scenario: Detect threshold violation
- **WHEN** SpO2 value is 85 and lower_threshold is 90
- **THEN** `VitalMeasurement.is_below_threshold` SHALL return `True`
- **AND** `check_threshold_violations()` SHALL include SpO2 in the violation list

#### Scenario: Handle missing vitals gracefully
- **WHEN** an event's `vitals/` group is missing some vital types
- **THEN** the corresponding fields in `VitalsData` SHALL be `None`
- **AND** parsing SHALL complete without error

### Requirement: PyTorch Dataset
The system SHALL provide a PyTorch-compatible Dataset class for loading ECG alarm events from multiple HDF5 files. The dataset SHALL support configurable inclusion of vitals and auxiliary signals. Each sample SHALL return a dictionary with `ecg` tensor of shape `[7, 2400]`, optional vitals dict, optional ppg/resp tensors, metadata, event_id, and patient_id.

#### Scenario: Iterate dataset samples
- **WHEN** a dataset is created from a directory containing 2 HDF5 files with 3 events each
- **THEN** `len(dataset)` SHALL return `6`
- **AND** `dataset[0]` SHALL return a dict with `"ecg"` key containing a tensor of shape `[7, 2400]`

#### Scenario: Load specific event by ID
- **WHEN** `get_event_by_id("PT1234", "event_1001")` is called
- **THEN** the specific event SHALL be returned as a sample dict

### Requirement: Phase 0 CLI Harness
The system SHALL provide a CLI script `scripts/cli_phase0.py` that accepts an HDF5 filepath and optional event_id, loads the data, and prints a JSON summary to stdout. When no event_id is given, it SHALL list all events. When an event_id is given, it SHALL print the full event data summary including ECG signal statistics, vitals values, and metadata.

#### Scenario: CLI lists events
- **WHEN** `python -m scripts.cli_phase0 data/samples/test.h5` is run without event_id
- **THEN** stdout SHALL contain a JSON object with keys `file_metadata` and `events` listing all event IDs

#### Scenario: CLI shows event detail
- **WHEN** `python -m scripts.cli_phase0 data/samples/test.h5 event_1001` is run
- **THEN** stdout SHALL contain a JSON object with `event_id`, `ecg_summary` (lead names, sample counts, min/max/mean per lead), `vitals_summary`, and `metadata`
