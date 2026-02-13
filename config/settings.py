"""Configuration management for ECG interpretation system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LoaderConfig:
    """Configuration for HDF5 data loading."""

    expected_leads: list[str] = field(
        default_factory=lambda: ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]
    )
    ecg_sample_rate: int = 200
    ecg_samples_per_lead: int = 2400
    ppg_sample_rate: float = 75.0
    ppg_samples: int = 900
    resp_sample_rate: float = 33.33
    resp_samples: int = 400
    strip_duration_sec: float = 12.0
    alarm_offset_seconds: float = 6.0


@dataclass
class PreprocessingConfig:
    """Configuration for signal preprocessing."""

    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    sqi_threshold: float = 0.3
    denoiser_weights: Optional[str] = None


@dataclass
class PipelineConfig:
    """Controls which pipeline components are enabled.

    When the heatmap model is untrained or fiducial/HR results are unreliable,
    individual measurement stages can be disabled while still running the
    arrhythmia classification (neural condition classifier).
    """

    enable_beat_analysis: bool = True
    """Enable beat detection, fiducial extraction, and per-beat analysis.
    When False, Phase 3 fiducial/beat pipeline is skipped entirely."""

    enable_heart_rate: bool = True
    """Enable heart rate calculation from R-peak intervals.
    Requires enable_beat_analysis=True to have effect."""

    enable_interval_measurements: bool = True
    """Enable PR, QRS, QT, QTc interval measurements.
    Requires enable_beat_analysis=True to have effect."""

    beat_detector: str = "signal"
    """Beat detection method: 'signal' (classical DSP) or 'heatmap' (neural).
    The 'signal' detector works without a trained heatmap model."""


@dataclass
class Settings:
    """Top-level application settings."""

    loader: LoaderConfig = field(default_factory=LoaderConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    data_dir: str = "data"
    model_weights_dir: str = "models/weights"
    log_level: str = "INFO"
    llm_provider: str = "anthropic"  # "anthropic" or "openai"
    llm_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables."""
        return cls(
            data_dir=os.getenv("ECG_DATA_DIR", "data"),
            model_weights_dir=os.getenv("ECG_MODEL_WEIGHTS_DIR", "models/weights"),
            log_level=os.getenv("ECG_LOG_LEVEL", "INFO"),
            llm_provider=os.getenv("ECG_LLM_PROVIDER", "anthropic"),
            llm_api_key=os.getenv("ECG_LLM_API_KEY"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> Settings:
        """Load settings from YAML config file."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        settings = cls()
        if "loader" in data:
            settings.loader = LoaderConfig(**data["loader"])
        if "preprocessing" in data:
            settings.preprocessing = PreprocessingConfig(**data["preprocessing"])
        if "pipeline" in data:
            settings.pipeline = PipelineConfig(**data["pipeline"])
        for key in ("data_dir", "model_weights_dir", "log_level", "llm_provider", "llm_api_key"):
            if key in data:
                setattr(settings, key, data[key])
        return settings
