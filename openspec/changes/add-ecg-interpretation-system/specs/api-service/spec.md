## ADDED Requirements

### Requirement: FastAPI Application
The system SHALL provide a FastAPI-based HTTP API that wraps the full ECG interpretation pipeline. The application SHALL be created via a factory function supporting configuration injection. The application SHALL include health check, analysis, and query endpoints. The API SHALL support JSON request/response with proper content-type headers.

#### Scenario: Application startup
- **WHEN** the FastAPI application is started via `uvicorn src.api.app:app`
- **THEN** the server SHALL start without error
- **AND** the health endpoint SHALL respond to requests

### Requirement: API Endpoints
The system SHALL expose the following HTTP endpoints:

1. `GET /health` — Returns service health status and model readiness
2. `POST /analyze` — Accepts HDF5 file path and event_id, runs full pipeline, returns JSON Feature Assembly
3. `POST /query` — Accepts HDF5 file path, event_id, and natural language query, returns query response

All endpoints SHALL return proper HTTP status codes (200 success, 400 bad request, 422 validation error, 500 internal error). Error responses SHALL include a JSON body with `error` and `detail` fields.

#### Scenario: Health check
- **WHEN** `GET /health` is called
- **THEN** response SHALL be `200 OK` with `{"status": "healthy", "models_loaded": true/false}`

#### Scenario: Analyze event
- **WHEN** `POST /analyze` is called with `{"hdf5_path": "/path/to/file.h5", "event_id": "event_1001"}`
- **THEN** response SHALL be `200 OK` with the complete JSON Feature Assembly as body
- **AND** `Content-Type` SHALL be `application/json`

#### Scenario: Query event
- **WHEN** `POST /query` is called with `{"hdf5_path": "/path/to/file.h5", "event_id": "event_1001", "query": "What is the heart rate?"}`
- **THEN** response SHALL be `200 OK` with the query response JSON

#### Scenario: Invalid file path
- **WHEN** `POST /analyze` is called with a non-existent HDF5 file path
- **THEN** response SHALL be `400 Bad Request` with `{"error": "file_not_found", "detail": "..."}`

#### Scenario: Invalid event ID
- **WHEN** `POST /analyze` is called with a valid file but non-existent event_id
- **THEN** response SHALL be `400 Bad Request` with `{"error": "event_not_found", "detail": "..."}`

### Requirement: Request/Response Schemas
The system SHALL define Pydantic models for all API request and response bodies. Request models SHALL validate required fields and types. Response models SHALL enforce the JSON Feature Assembly schema. The schemas SHALL be auto-documented via FastAPI's OpenAPI spec generation.

#### Scenario: Pydantic validation
- **WHEN** a `POST /analyze` request is missing the `event_id` field
- **THEN** response SHALL be `422 Unprocessable Entity` with Pydantic validation error details

### Requirement: API Middleware
The system SHALL include middleware for: structured request/response logging (including processing time), error handling (catch unhandled exceptions, return JSON error responses), and CORS configuration (configurable allowed origins).

#### Scenario: Request logging
- **WHEN** any API request is processed
- **THEN** the request method, path, status code, and processing time (ms) SHALL be logged

#### Scenario: Unhandled exception
- **WHEN** an unhandled exception occurs during request processing
- **THEN** the middleware SHALL catch it and return `500 Internal Server Error` with `{"error": "internal_error", "detail": "..."}`
- **AND** the exception SHALL be logged with full traceback

### Requirement: Main Pipeline Orchestration
The system SHALL provide a `Pipeline` class in `src/ecg_system/pipeline.py` that orchestrates all 6 stages (data loading → preprocessing → encoding → prediction → interpretation → query) into a single callable. The pipeline SHALL accept an HDF5 file path and event_id and return the JSON Feature Assembly. The pipeline SHALL optionally accept a query string and return a query response.

#### Scenario: Run full analysis pipeline
- **WHEN** `pipeline.analyze(hdf5_path, event_id)` is called
- **THEN** the pipeline SHALL execute all stages sequentially
- **AND** return the complete JSON Feature Assembly

#### Scenario: Run analysis with query
- **WHEN** `pipeline.query(hdf5_path, event_id, "What is the heart rate?")` is called
- **THEN** the pipeline SHALL run analysis then query
- **AND** return the query response

### Requirement: Docker Deployment
The system SHALL provide Docker configuration files: a `Dockerfile` for building the application image and a `docker-compose.yml` for local development. The Docker image SHALL include all dependencies and model weight mounting support.

#### Scenario: Docker build and run
- **WHEN** `docker compose up` is run
- **THEN** the API service SHALL start and be accessible on the configured port
- **AND** the health endpoint SHALL return `200 OK`
