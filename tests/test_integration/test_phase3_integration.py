"""Integration tests for Phase 3: features → heatmaps → fiducial points."""

import numpy as np
import torch
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.denoiser import ECGDenoiser
from src.encoding.foundation import FoundationModelAdapter
from src.prediction.heatmap import HeatmapDecoder
from src.prediction.gnn import LeadGNN
from src.prediction.fiducial import FiducialExtractor
from src.ecg_system.schemas import Beat

pytestmark = pytest.mark.integration


class TestPhase3Integration:
    """End-to-end: HDF5 → Phase 0 → Phase 1 → Phase 2 → Phase 3."""

    def test_full_pipeline(self, sample_hdf5_path: str) -> None:
        """Load → denoise → encode → decode → extract beats."""
        loader = HDF5AlarmEventLoader()
        denoiser = ECGDenoiser()
        denoiser.eval()
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()
        decoder = HeatmapDecoder(d_model=256)
        decoder.eval()
        extractor = FiducialExtractor()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            denoised = denoiser(ecg)
            features = encoder(denoised)
            heatmaps = decoder(features)

        hm_np = heatmaps.squeeze(0).numpy()
        beats = extractor.extract(hm_np, event.ecg.as_array)

        assert isinstance(beats, list)
        for beat in beats:
            assert isinstance(beat, Beat)
            assert isinstance(beat.fiducials, dict)

    def test_heatmap_shape_from_real_data(self, sample_hdf5_path: str) -> None:
        """Heatmaps from real data must be [7, 9, 2400]."""
        loader = HDF5AlarmEventLoader()
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()
        decoder = HeatmapDecoder(d_model=256)
        decoder.eval()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            features = encoder(ecg)
            heatmaps = decoder(features)

        assert heatmaps.shape == (1, 7, 9, 2400)
        assert heatmaps.min() >= 0.0
        assert heatmaps.max() <= 1.0

    def test_gnn_with_features(self, sample_hdf5_path: str) -> None:
        """GNN produces valid embeddings from encoded features."""
        loader = HDF5AlarmEventLoader()
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()
        gnn = LeadGNN(d_in=256, d_hidden=128)
        gnn.eval()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            features = encoder(ecg)  # [1, 7, S, D]
            # Mean pool over sequence for GNN input
            node_features = features.mean(dim=2)  # [1, 7, D]
            g_emb, n_emb = gnn(node_features)

        assert g_emb.shape == (1, 128)
        assert n_emb.shape == (1, 7, 128)
        assert torch.isfinite(g_emb).all()
        assert torch.isfinite(n_emb).all()

    def test_multiple_events(self, sample_hdf5_path: str) -> None:
        """Pipeline works on multiple events."""
        loader = HDF5AlarmEventLoader()
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()
        decoder = HeatmapDecoder(d_model=256)
        decoder.eval()
        extractor = FiducialExtractor()

        with loader.load_file(sample_hdf5_path) as f:
            for eid in loader.list_events(f):
                event = loader.load_event(f, eid)
                ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
                with torch.no_grad():
                    features = encoder(ecg)
                    heatmaps = decoder(features)
                beats = extractor.extract(heatmaps.squeeze(0).numpy())
                assert isinstance(beats, list)

    def test_beats_have_valid_samples(self, sample_hdf5_path: str) -> None:
        """All fiducial sample indices must be in [0, 2400)."""
        loader = HDF5AlarmEventLoader()
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()
        decoder = HeatmapDecoder(d_model=256)
        decoder.eval()
        extractor = FiducialExtractor()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            features = encoder(ecg)
            heatmaps = decoder(features)

        beats = extractor.extract(heatmaps.squeeze(0).numpy())
        for beat in beats:
            for name, fp in beat.fiducials.items():
                assert 0 <= fp.sample < 2400
                assert 0.0 <= fp.confidence <= 1.0
