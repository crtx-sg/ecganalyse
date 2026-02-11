# Tasks: Add ECG Interpretation System

## 0. Project Setup
- [x] 0.1 Initialize project structure (`pyproject.toml`, `src/`, `tests/`, `scripts/`, `config/`, `data/`, `docs/`, `docker/`)
- [x] 0.2 Create `requirements.txt` and `pyproject.toml` with all dependencies
- [x] 0.3 Configure linting and type checking (in `pyproject.toml` — ruff + mypy sections)
- [x] 0.4 Create `config/settings.py` with configuration management (env-based)
- [x] 0.5 Create `config/model_config.yaml` with model hyperparameters
- [x] 0.6 Create `config/rules_config.yaml` with clinical rules definitions
- [x] 0.7 Create `src/ecg_system/exceptions.py` with custom exception hierarchy
- [x] 0.8 Create `src/ecg_system/schemas.py` with all dataclasses (input + output)
- [x] 0.9 Create sample HDF5 test fixture (`data/samples/` + `tests/conftest.py`)
- [x] 0.10 Set up CI configuration (static checks: mypy + ruff in pyproject.toml)

## 1. Phase 0: Data Loading
- [x] 1.1 Implement `src/data/hdf5_loader.py` — `HDF5AlarmEventLoader` class (load_file, list_events, load_event, load_metadata)
- [x] 1.2 Implement `src/data/event_parser.py` — `AlarmEventParser` class (parse_ecg 7-lead, parse_ppg, parse_resp, parse_event)
- [x] 1.3 Implement `src/data/vitals_parser.py` — `VitalsParser` class (parse_vital, parse_all_vitals, check_threshold_violations)
- [x] 1.4 Implement `src/data/dataset.py` — `ECGAlarmDataset` PyTorch Dataset class
- [x] 1.5 Write unit tests: `tests/test_data/test_hdf5_loader.py`
- [x] 1.6 Write unit tests: `tests/test_data/test_event_parser.py`
- [x] 1.7 Write unit tests: `tests/test_data/test_vitals_parser.py`
- [x] 1.8 Write unit tests: `tests/test_data/test_dataset.py`
- [x] 1.9 Write gate tests: `tests/gates/test_phase0_gate.py` (AlarmEvent output contract: shapes, types, value ranges, required fields)
- [x] 1.10 Write CLI harness: `scripts/cli_phase0.py` — accepts HDF5 filepath + optional event_id, prints JSON summary of loaded data
- [x] 1.11 Write integration test: `tests/test_integration/test_phase0_integration.py` (full HDF5 → AlarmEvent round-trip)

## 2. Phase 1: Signal Preprocessing
- [ ] 2.1 Implement `src/preprocessing/quality.py` — `SignalQualityAssessor` (assess, compute_sqi, detect_lead_off, detect_saturation, check_pacer_presence)
- [ ] 2.2 Implement `src/preprocessing/denoiser.py` — `ECGDenoiser` U-Net model (input: [B,7,2400], output: [B,7,2400])
- [ ] 2.3 Implement `src/preprocessing/utils.py` — bandpass filter, baseline wander removal, normalization utilities
- [ ] 2.4 Write unit tests: `tests/test_preprocessing/test_quality.py`
- [ ] 2.5 Write unit tests: `tests/test_preprocessing/test_denoiser.py` (forward pass shape, residual learning)
- [ ] 2.6 Write unit tests: `tests/test_preprocessing/test_utils.py`
- [ ] 2.7 Write gate tests: `tests/gates/test_phase1_gate.py` (QualityReport contract + denoised tensor shape [7,2400])
- [ ] 2.8 Write CLI harness: `scripts/cli_phase1.py` — loads HDF5 event, runs quality + denoising, prints QualityReport JSON + signal stats
- [ ] 2.9 Write training script: `scripts/train_denoiser.py`
- [ ] 2.10 Write integration test: `tests/test_integration/test_phase1_integration.py` (AlarmEvent → QualityReport + denoised signals)

## 3. Phase 2: Feature Encoding
- [ ] 3.1 Implement `src/encoding/mamba.py` — `ECGMamba` model (input: [B,7,2400], output: [B,7,seq,d_model])
- [ ] 3.2 Implement `src/encoding/swin.py` — `Swin1DTransformer` model (input: [B,7,2400], output: [B,7,seq,embed_dim])
- [ ] 3.3 Implement `src/encoding/fusion.py` — `DualPathFusion` module (merge Mamba + Swin features)
- [ ] 3.4 Implement `src/encoding/foundation.py` — adapter for optional pretrained ECG foundation model
- [ ] 3.5 Write unit tests: `tests/test_encoding/test_mamba.py` (forward shape, gradient flow)
- [ ] 3.6 Write unit tests: `tests/test_encoding/test_swin.py` (forward shape, window attention)
- [ ] 3.7 Write unit tests: `tests/test_encoding/test_fusion.py` (merged output shape)
- [ ] 3.8 Write gate tests: `tests/gates/test_phase2_gate.py` (fused features tensor shape + dtype contract)
- [ ] 3.9 Write CLI harness: `scripts/cli_phase2.py` — loads event, preprocesses, encodes, prints feature tensor stats
- [ ] 3.10 Write training script: `scripts/train_encoder.py`
- [ ] 3.11 Write integration test: `tests/test_integration/test_phase2_integration.py` (denoised ECG → fused features)

## 4. Phase 3: Dense Prediction
- [ ] 4.1 Implement `src/prediction/heatmap.py` — U-Net decoder for heatmap regression (input: features, output: heatmaps per fiducial type)
- [ ] 4.2 Implement `src/prediction/gnn.py` — `LeadGNN` Graph Attention Network (7-node graph with anatomical adjacency)
- [ ] 4.3 Implement `src/prediction/fiducial.py` — fiducial point extraction from heatmaps (peak detection, confidence, beat segmentation)
- [ ] 4.4 Write unit tests: `tests/test_prediction/test_heatmap.py`
- [ ] 4.5 Write unit tests: `tests/test_prediction/test_gnn.py` (graph structure, attention weights, output shapes)
- [ ] 4.6 Write unit tests: `tests/test_prediction/test_fiducial.py` (peak detection accuracy, beat boundary logic)
- [ ] 4.7 Write gate tests: `tests/gates/test_phase3_gate.py` (List[Beat] output contract: required fiducials, confidence ranges, valid sample indices)
- [ ] 4.8 Write CLI harness: `scripts/cli_phase3.py` — runs full pipeline through prediction, prints detected beats + fiducial points as JSON
- [ ] 4.9 Write training script: `scripts/train_heatmap.py`
- [ ] 4.10 Write integration test: `tests/test_integration/test_phase3_integration.py` (features → fiducial points + beats)

## 5. Phase 4: Neuro-Symbolic Interpretation
- [ ] 5.1 Implement `src/interpretation/symbolic.py` — `SymbolicCalculationEngine` (compute intervals: PR, QRS, QT, QTc, RR; compute heart rate; all with calculation traces)
- [ ] 5.2 Implement `src/interpretation/rules.py` — `RuleBasedReasoningEngine` (rhythm classification, conduction abnormalities, morphology findings, ischemia markers)
- [ ] 5.3 Implement `src/interpretation/vitals_context.py` — `VitalsContextIntegrator` (HR validation, threshold checks, cross-modal findings)
- [ ] 5.4 Implement `src/interpretation/assembly.py` — `JSONAssembler` (assemble full JSON Feature Assembly v1.1)
- [ ] 5.5 Write unit tests: `tests/test_interpretation/test_symbolic.py` (interval calculations, traces, edge cases)
- [ ] 5.6 Write unit tests: `tests/test_interpretation/test_rules.py` (each rule fires correctly)
- [ ] 5.7 Write unit tests: `tests/test_interpretation/test_vitals_context.py` (threshold violations, HR validation)
- [ ] 5.8 Write unit tests: `tests/test_interpretation/test_assembly.py` (JSON schema compliance, required fields)
- [ ] 5.9 Write gate tests: `tests/gates/test_phase4_gate.py` (JSON Assembly output contract: schema version, required sections, trace presence)
- [ ] 5.10 Write CLI harness: `scripts/cli_phase4.py` — runs full pipeline through interpretation, prints complete JSON Feature Assembly
- [ ] 5.11 Write integration test: `tests/test_integration/test_phase4_integration.py` (fiducials + vitals → JSON Assembly)

## 6. Phase 5: Clinical Query Interface
- [ ] 6.1 Implement `src/query/router.py` — `QueryRouter` (classify queries as Type A deterministic vs Type B LLM-required)
- [ ] 6.2 Implement `src/query/deterministic.py` — `DeterministicHandler` (answer Type A queries: measurements, vitals, findings lookup)
- [ ] 6.3 Implement `src/query/llm_handler.py` — `LLMHandler` (answer Type B queries: clinical reasoning, differential diagnosis, correlation)
- [ ] 6.4 Implement `src/query/prompts.py` — LLM prompt templates (system prompt with JSON Assembly, clinical guardrails)
- [ ] 6.5 Write unit tests: `tests/test_query/test_router.py` (query classification accuracy)
- [ ] 6.6 Write unit tests: `tests/test_query/test_deterministic.py` (correct answers for known queries)
- [ ] 6.7 Write unit tests: `tests/test_query/test_llm_handler.py` (prompt construction, response parsing; mock LLM)
- [ ] 6.8 Write gate tests: `tests/gates/test_phase5_gate.py` (query response contract: answer text, confidence, sources, trace)
- [ ] 6.9 Write CLI harness: `scripts/cli_phase5.py` — runs full pipeline, accepts natural language query, prints answer
- [ ] 6.10 Write integration test: `tests/test_integration/test_phase5_integration.py` (JSON Assembly + query → answer)

## 7. Phase 6: API Service & Integration
- [ ] 7.1 Implement `src/api/app.py` — FastAPI application factory
- [ ] 7.2 Implement `src/api/routes.py` — API endpoints (POST /analyze, POST /query, GET /health)
- [ ] 7.3 Implement `src/api/schemas.py` — Pydantic request/response models
- [ ] 7.4 Implement `src/api/middleware.py` — request logging, error handling, CORS
- [ ] 7.5 Implement `src/ecg_system/pipeline.py` — main orchestration tying all stages together
- [ ] 7.6 Write unit tests: `tests/test_api/test_routes.py` (endpoint behavior with mocked pipeline)
- [ ] 7.7 Write unit tests: `tests/test_api/test_schemas.py` (Pydantic validation)
- [ ] 7.8 Write gate tests: `tests/gates/test_phase6_gate.py` (API response contract: status codes, response schema, content-type)
- [ ] 7.9 Write integration test: `tests/test_integration/test_phase6_integration.py` (HTTP request → full pipeline → response)
- [ ] 7.10 Write end-to-end integration test: `tests/test_integration/test_e2e.py` (HDF5 file → API → query answer)
- [ ] 7.11 Create `docker/Dockerfile` and `docker/docker-compose.yml`
- [ ] 7.12 Write documentation: `docs/api.md`, `docs/architecture.md`, `docs/data_format.md`, `docs/deployment.md`
