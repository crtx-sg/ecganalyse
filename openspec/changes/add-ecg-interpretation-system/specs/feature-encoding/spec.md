## ADDED Requirements

### Requirement: ECG-Mamba Encoder
The system SHALL provide an ECG-Mamba encoder based on Selective State Space Models that processes 7-lead ECG input of shape `[batch, 7, 2400]` and outputs per-lead temporal features of shape `[batch, 7, seq_len, d_model]`. The encoder SHALL capture long-range temporal dependencies (rhythm patterns, RR interval regularity) across the full 12-second strip. Configuration SHALL include `d_model=256`, `d_state=64`, `n_layers=4` as defaults.

#### Scenario: Forward pass shape validation
- **WHEN** a tensor of shape `[2, 7, 2400]` is passed through ECGMamba with default config
- **THEN** the output SHALL have shape `[2, 7, seq_len, 256]` where `seq_len` depends on the patching/tokenization scheme
- **AND** gradients SHALL flow back through the model (no gradient breakage)

#### Scenario: Per-lead processing
- **WHEN** the encoder processes 7 leads
- **THEN** each lead SHALL be encoded independently (shared weights across leads)
- **AND** the output SHALL maintain lead ordering: ECG1, ECG2, ECG3, aVR, aVL, aVF, vVX

### Requirement: 1D-Swin Transformer Encoder
The system SHALL provide a hierarchical 1D Swin Transformer that processes 7-lead ECG input of shape `[batch, 7, 2400]` and outputs per-lead morphological features of shape `[batch, 7, seq_len, embed_dim]`. The transformer SHALL use shifted window attention with 4 hierarchical levels optimized for 200Hz ECG: Level 1 (20 samples / 100ms — wave morphology), Level 2 (100 samples / 500ms — single beat), Level 3 (400 samples / 2s — beat-to-beat), Level 4 (full strip — global rhythm).

#### Scenario: Forward pass shape validation
- **WHEN** a tensor of shape `[2, 7, 2400]` is passed through Swin1DTransformer
- **THEN** the output SHALL have shape `[2, 7, seq_len, embed_dim]`
- **AND** gradients SHALL flow without breakage

#### Scenario: Hierarchical window sizes
- **WHEN** the model is initialized with default config
- **THEN** the 4 levels SHALL use window sizes of 20, 100, 400, and 2400 samples respectively

### Requirement: Dual-Path Feature Fusion
The system SHALL fuse features from ECG-Mamba and 1D-Swin Transformer into a single unified representation. The fusion module SHALL accept both encoder outputs and produce a fused feature tensor. The fusion strategy SHALL combine global context (Mamba) with local morphology (Swin) through concatenation and learned projection (or cross-attention if configured).

#### Scenario: Fuse Mamba and Swin features
- **WHEN** Mamba outputs shape `[B, 7, S1, D1]` and Swin outputs shape `[B, 7, S2, D2]`
- **THEN** the fusion module SHALL produce output of shape `[B, 7, S_fused, D_fused]`
- **AND** `D_fused` and `S_fused` SHALL be deterministic for given config

### Requirement: Foundation Model Adapter
The system SHALL provide an optional adapter module that can wrap a pretrained ECG foundation model encoder as a drop-in replacement for the Mamba+Swin dual path. When no foundation model is available, the system SHALL fall back to the dual-path encoding.

#### Scenario: Fallback to dual-path
- **WHEN** no foundation model weights are provided
- **THEN** the system SHALL use ECGMamba + Swin1DTransformer + DualPathFusion
- **AND** no errors SHALL occur

### Requirement: Phase 2 CLI Harness
The system SHALL provide a CLI script `scripts/cli_phase2.py` that loads an HDF5 event, preprocesses it, encodes it, and prints feature tensor statistics (shape, dtype, min, max, mean, std per lead) to stdout as JSON.

#### Scenario: CLI runs Phase 2 pipeline
- **WHEN** `python -m scripts.cli_phase2 data/samples/test.h5 event_1001` is run
- **THEN** stdout SHALL contain a JSON object with `encoder_output` containing shape, dtype, and per-lead statistics
