"""Integration tests for Phase 1: AlarmEvent → QualityReport + denoised tensor."""

import numpy as np
import torch
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.quality import SignalQualityAssessor
from src.preprocessing.denoiser import ECGDenoiser
from src.preprocessing.utils import bandpass_filter, normalize_leads
from src.ecg_system.schemas import QualityReport

pytestmark = pytest.mark.integration


class TestPhase1Integration:
    """End-to-end: HDF5 → Phase 0 loader → quality + denoising → validate."""

    def test_full_pipeline(self, sample_hdf5_path: str) -> None:
        """Load event → assess quality → denoise → verify types and shapes."""
        loader = HDF5AlarmEventLoader()
        assessor = SignalQualityAssessor()
        denoiser = ECGDenoiser()
        denoiser.eval()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        # Quality assessment
        report = assessor.assess(event.ecg)
        assert isinstance(report, QualityReport)
        assert 0.0 <= report.overall_sqi <= 1.0
        assert len(report.lead_sqi) == 7

        # Denoising
        ecg_array = event.ecg.as_array
        assert ecg_array.shape == (7, 2400)

        x = torch.from_numpy(ecg_array).unsqueeze(0).float()
        with torch.no_grad():
            y = denoiser(x)

        denoised = y.squeeze(0).numpy()
        assert denoised.shape == (7, 2400)
        assert denoised.dtype == np.float32

    def test_quality_then_filter(self, sample_hdf5_path: str) -> None:
        """Quality assessment followed by bandpass filtering on each lead."""
        loader = HDF5AlarmEventLoader()
        assessor = SignalQualityAssessor()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        report = assessor.assess(event.ecg)

        # Filter only usable leads
        for lead in report.usable_leads:
            sig = event.ecg.signals[lead]
            filtered = bandpass_filter(sig, event.ecg.sample_rate)
            assert filtered.shape == sig.shape
            assert filtered.dtype == sig.dtype

    def test_normalize_after_denoise(self, sample_hdf5_path: str) -> None:
        """Denoise → normalize → verify zero-mean unit-var."""
        loader = HDF5AlarmEventLoader()
        denoiser = ECGDenoiser()
        denoiser.eval()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        ecg_array = event.ecg.as_array
        x = torch.from_numpy(ecg_array).unsqueeze(0).float()
        with torch.no_grad():
            y = denoiser(x)
        denoised = y.squeeze(0).numpy()

        normed = normalize_leads(denoised)
        assert normed.shape == (7, 2400)
        for i in range(7):
            assert abs(np.mean(normed[i])) < 0.01
            assert abs(np.std(normed[i]) - 1.0) < 0.05

    def test_multiple_events(self, sample_hdf5_path: str) -> None:
        """Pipeline works on multiple events from the same file."""
        loader = HDF5AlarmEventLoader()
        assessor = SignalQualityAssessor()

        with loader.load_file(sample_hdf5_path) as f:
            event_ids = loader.list_events(f)
            assert len(event_ids) >= 2

            reports = []
            for eid in event_ids:
                event = loader.load_event(f, eid)
                report = assessor.assess(event.ecg)
                reports.append(report)

        # Each event produces a valid report
        for report in reports:
            assert isinstance(report, QualityReport)
            assert 0.0 <= report.overall_sqi <= 1.0

    def test_denoiser_preserves_energy(self, sample_hdf5_path: str) -> None:
        """Denoised signal RMS should be within 50% of original."""
        loader = HDF5AlarmEventLoader()
        denoiser = ECGDenoiser()
        denoiser.eval()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        ecg_array = event.ecg.as_array
        x = torch.from_numpy(ecg_array).unsqueeze(0).float()
        with torch.no_grad():
            y = denoiser(x)
        denoised = y.squeeze(0).numpy()

        rms_in = np.sqrt(np.mean(ecg_array**2))
        rms_out = np.sqrt(np.mean(denoised**2))
        assert rms_out > 0.5 * rms_in, f"Energy too low: {rms_out:.4f} vs {rms_in:.4f}"
