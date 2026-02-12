"""Gate tests for Phase 2: Feature Encoding output contracts."""

import numpy as np
import torch
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.encoding.foundation import FoundationModelAdapter

pytestmark = pytest.mark.gate


class TestFeatureEncodingContract:
    """Validate fused feature tensor output matches Phase 2 contract."""

    def setup_method(self) -> None:
        self.loader = HDF5AlarmEventLoader()
        self.encoder = FoundationModelAdapter(output_dim=256)
        self.encoder.eval()

    def test_fused_shape(self, sample_hdf5_path: str) -> None:
        """Fused tensor must be [7, seq_len, 256]."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            out = self.encoder(ecg)
        features = out.squeeze(0)  # [7, S, D]
        assert features.ndim == 3
        assert features.shape[0] == 7
        assert features.shape[2] == 256

    def test_fused_dtype(self, sample_hdf5_path: str) -> None:
        """Fused tensor must be float32."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            out = self.encoder(ecg)
        assert out.dtype == torch.float32

    def test_fused_finite(self, sample_hdf5_path: str) -> None:
        """All values must be finite (no NaN/Inf)."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            out = self.encoder(ecg)
        assert torch.isfinite(out).all()

    def test_fused_deterministic_shape(self, sample_hdf5_path: str) -> None:
        """Shape must be deterministic across runs."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            out1 = self.encoder(ecg)
            out2 = self.encoder(ecg)
        assert out1.shape == out2.shape

    def test_seven_leads_preserved(self, sample_hdf5_path: str) -> None:
        """Lead dimension must be preserved at 7."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            out = self.encoder(ecg)
        assert out.shape[1] == 7
