"""Gate tests for Phase 3: Dense Prediction output contracts."""

import numpy as np
import torch
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.encoding.foundation import FoundationModelAdapter
from src.prediction.heatmap import HeatmapDecoder, NUM_FIDUCIAL_TYPES
from src.prediction.fiducial import FiducialExtractor
from src.ecg_system.schemas import Beat, FiducialPoint

pytestmark = pytest.mark.gate

VALID_BEAT_TYPES = {"normal", "pvc", "pac", "paced", "unclassified"}


class TestHeatmapOutputContract:
    """Validate heatmap decoder output matches Phase 3 contract."""

    def setup_method(self) -> None:
        self.loader = HDF5AlarmEventLoader()
        self.encoder = FoundationModelAdapter(output_dim=256)
        self.encoder.eval()
        self.decoder = HeatmapDecoder(d_model=256)
        self.decoder.eval()

    def test_heatmap_shape(self, sample_hdf5_path: str) -> None:
        """Heatmaps must be [7, 9, 2400]."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            features = self.encoder(ecg)
            heatmaps = self.decoder(features)

        hm = heatmaps.squeeze(0)
        assert hm.shape == (7, 9, 2400)

    def test_heatmap_range(self, sample_hdf5_path: str) -> None:
        """All heatmap values must be in [0, 1]."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            features = self.encoder(ecg)
            heatmaps = self.decoder(features)

        assert heatmaps.min() >= 0.0
        assert heatmaps.max() <= 1.0


class TestBeatOutputContract:
    """Validate Beat output matches Phase 3 contract."""

    def setup_method(self) -> None:
        self.loader = HDF5AlarmEventLoader()
        self.encoder = FoundationModelAdapter(output_dim=256)
        self.encoder.eval()
        self.decoder = HeatmapDecoder(d_model=256)
        self.decoder.eval()
        self.extractor = FiducialExtractor()

    def _get_beats(self, sample_hdf5_path: str) -> list[Beat]:
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")

        ecg = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            features = self.encoder(ecg)
            heatmaps = self.decoder(features)

        hm_np = heatmaps.squeeze(0).numpy()
        return self.extractor.extract(hm_np, event.ecg.as_array)

    def test_beat_type_valid(self, sample_hdf5_path: str) -> None:
        """Every beat_type must be one of the valid types."""
        beats = self._get_beats(sample_hdf5_path)
        for beat in beats:
            assert beat.beat_type in VALID_BEAT_TYPES, (
                f"Invalid beat_type: {beat.beat_type}"
            )

    def test_fiducial_samples_in_range(self, sample_hdf5_path: str) -> None:
        """All fiducial sample indices must be in [0, 2400)."""
        beats = self._get_beats(sample_hdf5_path)
        for beat in beats:
            for name, fp in beat.fiducials.items():
                assert 0 <= fp.sample < 2400, (
                    f"Beat {beat.beat_index}, {name}: sample={fp.sample} out of range"
                )

    def test_fiducial_confidence_range(self, sample_hdf5_path: str) -> None:
        """All confidence scores must be in [0, 1]."""
        beats = self._get_beats(sample_hdf5_path)
        for beat in beats:
            for name, fp in beat.fiducials.items():
                assert 0.0 <= fp.confidence <= 1.0, (
                    f"Beat {beat.beat_index}, {name}: confidence={fp.confidence}"
                )

    def test_beat_has_required_fields(self, sample_hdf5_path: str) -> None:
        """Each Beat must have all required fields."""
        beats = self._get_beats(sample_hdf5_path)
        for beat in beats:
            assert isinstance(beat.beat_index, int)
            assert isinstance(beat.beat_type, str)
            assert isinstance(beat.lead, str)
            assert isinstance(beat.fiducials, dict)
            assert isinstance(beat.intervals, dict)
            assert isinstance(beat.morphology, dict)
            assert isinstance(beat.anomalies, list)
            assert isinstance(beat.anomaly_confidence, float)

    def test_beat_index_sequential(self, sample_hdf5_path: str) -> None:
        """Beat indices must be sequential starting from 0."""
        beats = self._get_beats(sample_hdf5_path)
        for i, beat in enumerate(beats):
            assert beat.beat_index == i
