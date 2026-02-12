"""Gate tests for Phase 1: Signal Preprocessing output contracts."""

import numpy as np
import torch
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.quality import SignalQualityAssessor
from src.preprocessing.denoiser import ECGDenoiser

pytestmark = pytest.mark.gate

EXPECTED_LEADS = {"ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"}


class TestQualityReportContract:
    """Validate QualityReport output matches Phase 1 contract."""

    def setup_method(self) -> None:
        self.loader = HDF5AlarmEventLoader()
        self.assessor = SignalQualityAssessor()

    def test_overall_sqi_range(self, sample_hdf5_path: str) -> None:
        """overall_sqi must be in [0, 1]."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        report = self.assessor.assess(event.ecg)
        assert 0.0 <= report.overall_sqi <= 1.0

    def test_lead_sqi_keys_and_range(self, sample_hdf5_path: str) -> None:
        """lead_sqi must have 7 keys, each in [0, 1]."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        report = self.assessor.assess(event.ecg)
        assert set(report.lead_sqi.keys()) == EXPECTED_LEADS
        for lead, sqi in report.lead_sqi.items():
            assert 0.0 <= sqi <= 1.0, f"{lead} SQI out of range: {sqi}"

    def test_usable_leads_subset(self, sample_hdf5_path: str) -> None:
        """usable_leads must be a subset of all leads."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        report = self.assessor.assess(event.ecg)
        assert set(report.usable_leads) <= EXPECTED_LEADS

    def test_noise_level_valid(self, sample_hdf5_path: str) -> None:
        """noise_level must be one of {low, moderate, high}."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        report = self.assessor.assess(event.ecg)
        assert report.noise_level in {"low", "moderate", "high"}

    def test_baseline_stability_valid(self, sample_hdf5_path: str) -> None:
        """baseline_stability must be one of {stable, moderate, unstable}."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        report = self.assessor.assess(event.ecg)
        assert report.baseline_stability in {"stable", "moderate", "unstable"}


class TestDenoisedTensorContract:
    """Validate denoised tensor output matches Phase 1 contract."""

    def setup_method(self) -> None:
        self.loader = HDF5AlarmEventLoader()
        self.denoiser = ECGDenoiser()
        self.denoiser.eval()

    def test_denoised_shape(self, sample_hdf5_path: str) -> None:
        """Denoised tensor must be [7, 2400]."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg_array = event.ecg.as_array  # [7, 2400]
        x = torch.from_numpy(ecg_array).unsqueeze(0).float()  # [1, 7, 2400]
        with torch.no_grad():
            y = self.denoiser(x)

        denoised = y.squeeze(0).numpy()
        assert denoised.shape == (7, 2400)

    def test_denoised_dtype(self, sample_hdf5_path: str) -> None:
        """Denoised tensor must be float32."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg_array = event.ecg.as_array
        x = torch.from_numpy(ecg_array).unsqueeze(0).float()
        with torch.no_grad():
            y = self.denoiser(x)

        denoised = y.squeeze(0).numpy()
        assert denoised.dtype == np.float32

    def test_denoised_value_range(self, sample_hdf5_path: str) -> None:
        """Denoised values must be in [-10, 10] (physiological range)."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg_array = event.ecg.as_array
        x = torch.from_numpy(ecg_array).unsqueeze(0).float()
        with torch.no_grad():
            y = self.denoiser(x)

        denoised = y.squeeze(0).numpy()
        assert denoised.min() >= -10.0, f"Min value {denoised.min()} out of range"
        assert denoised.max() <= 10.0, f"Max value {denoised.max()} out of range"
