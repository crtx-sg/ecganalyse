## ADDED Requirements

### Requirement: Heatmap Decoder
The system SHALL provide a U-Net decoder that takes encoded features and produces heatmap regression outputs for each fiducial point type (P-onset, P-peak, P-offset, QRS-onset, R-peak, QRS-offset, T-onset, T-peak, T-offset). Each heatmap SHALL have the same temporal resolution as the input signal (2400 samples) with values in [0, 1] representing fiducial point probability. The decoder SHALL produce heatmaps for all 7 leads.

#### Scenario: Heatmap output shape
- **WHEN** encoded features are passed through the heatmap decoder
- **THEN** the output SHALL have shape `[batch, 7, 9, 2400]` (7 leads x 9 fiducial types x 2400 time steps)
- **AND** all values SHALL be in range [0.0, 1.0]

#### Scenario: R-peak heatmap for regular rhythm
- **WHEN** a regular sinus rhythm ECG is processed
- **THEN** the R-peak heatmap SHALL show distinct peaks at each R-peak location
- **AND** the number of peaks SHALL correspond to the expected heart rate

### Requirement: Graph Attention Network for Multi-Lead Fusion
The system SHALL provide a Graph Attention Network (GAT) that models spatial relationships between the 7 ECG leads. The graph SHALL have 7 nodes (one per lead) with anatomically-informed edges: ECG1↔ECG2, ECG1↔aVL, ECG2↔ECG3, ECG2↔aVF, ECG3↔aVF, aVR↔aVL, aVR↔aVF, vVX↔ECG1, vVX↔ECG2. The GAT SHALL produce both a graph-level embedding and per-node (per-lead) enhanced embeddings.

#### Scenario: Graph structure validation
- **WHEN** the LeadGNN is initialized
- **THEN** the graph SHALL have exactly 7 nodes
- **AND** edges SHALL follow the anatomical adjacency defined in `LEAD_ADJACENCY`
- **AND** attention weights SHALL be learnable

#### Scenario: Forward pass
- **WHEN** per-lead embeddings of shape `[batch, 7, d_features]` are input
- **THEN** `graph_embedding` SHALL have shape `[batch, d_graph]`
- **AND** `node_embeddings` SHALL have shape `[batch, 7, d_node]`

### Requirement: Fiducial Point Extraction
The system SHALL extract fiducial points from heatmap outputs by applying peak detection with configurable thresholds. For each detected beat, the system SHALL extract: P-onset, P-peak, P-offset, QRS-onset, R-peak, QRS-offset, T-onset, T-peak, T-offset. Each fiducial point SHALL include sample index, time in milliseconds, and confidence score. The system SHALL segment beats using R-peak locations and validate physiological plausibility of detected fiducials.

#### Scenario: Extract fiducials from clean ECG
- **WHEN** heatmaps from a clean normal sinus rhythm ECG are processed
- **THEN** each detected beat SHALL have at least R-peak with confidence > 0.8
- **AND** fiducial time ordering SHALL be: P-onset < P-peak < P-offset < QRS-onset < R-peak < QRS-offset < T-onset < T-peak < T-offset

#### Scenario: Handle missing P-wave
- **WHEN** a beat has no detectable P-wave (e.g., atrial fibrillation)
- **THEN** P-onset, P-peak, P-offset SHALL be absent from that beat's fiducials
- **AND** QRS and T-wave fiducials SHALL still be extracted

#### Scenario: Beat classification
- **WHEN** fiducials are extracted for all beats
- **THEN** each beat SHALL be assigned a `beat_type`: one of `"normal"`, `"pvc"`, `"pac"`, `"paced"`, or `"unclassified"`

### Requirement: Phase 3 CLI Harness
The system SHALL provide a CLI script `scripts/cli_phase3.py` that runs the full pipeline through dense prediction and prints detected beats with fiducial points as JSON to stdout.

#### Scenario: CLI runs Phase 3 pipeline
- **WHEN** `python -m scripts.cli_phase3 data/samples/test.h5 event_1001` is run
- **THEN** stdout SHALL contain a JSON object with `beats` array, each beat containing `beat_index`, `beat_type`, `lead`, `fiducials` dict, and `confidence`
