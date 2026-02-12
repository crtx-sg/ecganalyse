"""Integration tests for Phase 2: denoised ECG → fused features."""

import numpy as np
import torch
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.quality import SignalQualityAssessor
from src.preprocessing.denoiser import ECGDenoiser
from src.preprocessing.utils import normalize_leads
from src.encoding.mamba import ECGMamba
from src.encoding.swin import Swin1DTransformer
from src.encoding.fusion import DualPathFusion
from src.encoding.foundation import FoundationModelAdapter

pytestmark = pytest.mark.integration


class TestPhase2Integration:
    """End-to-end: HDF5 → Phase 0 → Phase 1 → Phase 2 → validate."""

    def test_full_pipeline(self, sample_hdf5_path: str) -> None:
        """Load → denoise → encode → verify shapes."""
        loader = HDF5AlarmEventLoader()
        denoiser = ECGDenoiser()
        denoiser.eval()
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            denoised = denoiser(ecg)
            features = encoder(denoised)

        assert features.shape[0] == 1
        assert features.shape[1] == 7
        assert features.shape[3] == 256
        assert features.dtype == torch.float32

    def test_quality_gates_encoding(self, sample_hdf5_path: str) -> None:
        """Quality assessment followed by encoding on usable leads."""
        loader = HDF5AlarmEventLoader()
        assessor = SignalQualityAssessor()
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        report = assessor.assess(event.ecg)
        assert len(report.usable_leads) > 0

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            features = encoder(ecg)
        assert torch.isfinite(features).all()

    def test_normalize_then_encode(self, sample_hdf5_path: str) -> None:
        """Normalize → encode pipeline produces valid features."""
        loader = HDF5AlarmEventLoader()
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        normed = normalize_leads(event.ecg.as_array)
        x = torch.from_numpy(normed).unsqueeze(0).float()
        with torch.no_grad():
            features = encoder(x)
        assert features.shape[1] == 7
        assert torch.isfinite(features).all()

    def test_multiple_events(self, sample_hdf5_path: str) -> None:
        """Pipeline works on multiple events from the same file."""
        loader = HDF5AlarmEventLoader()
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()

        with loader.load_file(sample_hdf5_path) as f:
            event_ids = loader.list_events(f)
            shapes = []
            for eid in event_ids:
                event = loader.load_event(f, eid)
                ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
                with torch.no_grad():
                    features = encoder(ecg)
                shapes.append(features.shape)

        # All events should produce same shape
        for s in shapes:
            assert s == shapes[0]

    def test_individual_encoders(self, sample_hdf5_path: str) -> None:
        """Test Mamba and Swin individually before fusion."""
        loader = HDF5AlarmEventLoader()
        mamba = ECGMamba(d_model=256, d_state=64, n_layers=2, patch_size=10)
        swin = Swin1DTransformer()
        mamba.eval()
        swin.eval()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()

        with torch.no_grad():
            m_out = mamba(ecg)
            s_out = swin(ecg)

        assert m_out.shape[1] == 7
        assert s_out.shape[1] == 7
        assert torch.isfinite(m_out).all()
        assert torch.isfinite(s_out).all()

        # Fuse
        fusion = DualPathFusion(
            mamba_dim=256, swin_dim=swin.output_dim, output_dim=256,
        )
        fusion.eval()
        with torch.no_grad():
            fused = fusion(m_out, s_out)
        assert fused.shape[1] == 7
        assert fused.shape[3] == 256
