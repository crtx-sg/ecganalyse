"""Custom exception hierarchy for ECG interpretation system."""


class ECGSystemError(Exception):
    """Base exception for all ECG system errors."""


class HDF5LoadError(ECGSystemError):
    """Raised when an HDF5 file cannot be loaded or is corrupt."""


class EventParseError(ECGSystemError):
    """Raised when an alarm event cannot be parsed from HDF5."""

    def __init__(self, event_id: str, detail: str) -> None:
        self.event_id = event_id
        self.detail = detail
        super().__init__(f"Failed to parse event '{event_id}': {detail}")


class VitalsParseError(ECGSystemError):
    """Raised when vital signs cannot be parsed."""


class SignalQualityError(ECGSystemError):
    """Raised when signal quality is too low for analysis."""


class ModelNotLoadedError(ECGSystemError):
    """Raised when a required model is not loaded."""


class PipelineError(ECGSystemError):
    """Raised when the pipeline encounters an error during execution."""

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f"Pipeline error at stage '{stage}': {detail}")


class QueryError(ECGSystemError):
    """Raised when a clinical query cannot be processed."""
