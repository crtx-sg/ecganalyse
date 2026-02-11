## ADDED Requirements

### Requirement: Query Router
The system SHALL classify incoming clinical queries into two types: Type A (deterministic — answerable directly from JSON Assembly fields without LLM) and Type B (requires LLM reasoning). Type A queries include direct measurement lookups ("What is the heart rate?"), vitals checks ("Is SpO2 normal?"), and finding summaries ("List all abnormalities"). Type B queries include clinical reasoning ("Is this pattern consistent with ischemia?"), differential diagnosis ("What could cause this QTc prolongation?"), and complex correlation ("How do the vitals relate to the ECG findings?"). The router SHALL use keyword matching and pattern classification (not an LLM call) for routing.

#### Scenario: Route Type A query
- **WHEN** query is "What is the heart rate?"
- **THEN** the router SHALL classify it as `Type A`
- **AND** route to `DeterministicHandler`

#### Scenario: Route Type B query
- **WHEN** query is "Could this ECG pattern indicate acute coronary syndrome?"
- **THEN** the router SHALL classify it as `Type B`
- **AND** route to `LLMHandler`

#### Scenario: Route ambiguous query
- **WHEN** query is "Tell me about the QTc"
- **THEN** the router SHALL default to `Type A` if the field exists in JSON Assembly
- **AND** fall back to `Type B` only if deterministic answer is insufficient

### Requirement: Deterministic Query Handler
The system SHALL answer Type A queries by extracting values directly from the JSON Feature Assembly. Responses SHALL include the answer value, units, source field path in the JSON, and relevant calculation trace. The handler SHALL support queries about: any global measurement, any vital sign value, quality scores, rhythm classification, finding lookup, and beat count.

#### Scenario: Answer heart rate query
- **WHEN** query is "What is the heart rate?" and JSON Assembly contains `global_measurements.heart_rate_bpm = 72`
- **THEN** the response SHALL be: `{"answer": "The heart rate is 72 bpm.", "value": 72, "units": "bpm", "source": "global_measurements.heart_rate_bpm", "trace": "60000 / 832 = 72.1 bpm"}`

#### Scenario: Answer vital sign query
- **WHEN** query is "What is the SpO2 level?" and vitals_context contains `spo2.value = 96`
- **THEN** the response SHALL include the value (96), units (%), and threshold status

#### Scenario: Answer findings query
- **WHEN** query is "Are there any abnormalities?"
- **THEN** the response SHALL list all findings with severity > "normal"
- **AND** if none exist, SHALL respond "No abnormalities detected"

### Requirement: LLM Query Handler
The system SHALL answer Type B queries by sending the JSON Feature Assembly and the clinical query to an LLM (Anthropic Claude or OpenAI GPT) via API. The system SHALL construct a prompt that includes the full JSON Assembly as context, clinical guardrails (the LLM SHALL NOT fabricate measurements not in the assembly), and instructions to reference specific findings and traces in its answer. The response SHALL include the LLM's answer, a confidence indicator, and list of referenced findings/measurements.

#### Scenario: Answer clinical reasoning query
- **WHEN** query is "Is this pattern consistent with atrial fibrillation?" and JSON Assembly shows irregular RR intervals with absent P-waves
- **THEN** the LLM handler SHALL send the full JSON Assembly + query to the LLM
- **AND** the response SHALL reference specific measurements from the assembly

#### Scenario: Clinical guardrails enforced
- **WHEN** the LLM is prompted
- **THEN** the system prompt SHALL instruct: "Only reference measurements present in the provided JSON. Do not fabricate or estimate values not explicitly provided."

#### Scenario: LLM API failure
- **WHEN** the LLM API call fails (timeout, rate limit, etc.)
- **THEN** the handler SHALL return an error response with `"error": "LLM service unavailable"` and SHALL NOT crash

### Requirement: LLM Prompt Templates
The system SHALL maintain prompt templates in `src/query/prompts.py` including a system prompt (clinical context, guardrails, JSON schema explanation) and a user prompt template (JSON Assembly + query). Templates SHALL be versioned and testable independently. The system prompt SHALL instruct the LLM to include vitals context in its clinical reasoning.

#### Scenario: System prompt includes guardrails
- **WHEN** the system prompt template is rendered
- **THEN** it SHALL contain instructions about not fabricating measurements
- **AND** it SHALL reference the JSON Assembly schema version
- **AND** it SHALL instruct the LLM to consider vitals context

### Requirement: Phase 5 CLI Harness
The system SHALL provide a CLI script `scripts/cli_phase5.py` that runs the full pipeline, accepts a natural language query as a command-line argument, and prints the query response to stdout as JSON.

#### Scenario: CLI answers deterministic query
- **WHEN** `python -m scripts.cli_phase5 data/samples/test.h5 event_1001 "What is the heart rate?"` is run
- **THEN** stdout SHALL contain a JSON response with the heart rate answer, source, and trace

#### Scenario: CLI answers LLM query
- **WHEN** `python -m scripts.cli_phase5 data/samples/test.h5 event_1001 "Is this ECG concerning?"` is run
- **THEN** stdout SHALL contain a JSON response with the LLM's clinical reasoning answer
