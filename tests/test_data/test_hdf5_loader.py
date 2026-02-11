"""Unit tests for HDF5AlarmEventLoader."""

import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.ecg_system.exceptions import HDF5LoadError, EventParseError


class TestHDF5AlarmEventLoader:
    def setup_method(self) -> None:
        self.loader = HDF5AlarmEventLoader()

    def test_load_valid_file(self, sample_hdf5_path: str) -> None:
        f = self.loader.load_file(sample_hdf5_path)
        assert f is not None
        f.close()

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(HDF5LoadError, match="File not found"):
            self.loader.load_file("/nonexistent/path.h5")

    def test_load_wrong_extension(self, tmp_path) -> None:
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not hdf5")
        with pytest.raises(HDF5LoadError, match="Expected .h5 file"):
            self.loader.load_file(str(bad_file))

    def test_list_events(self, sample_hdf5_path: str) -> None:
        with self.loader.load_file(sample_hdf5_path) as f:
            events = self.loader.list_events(f)
        assert events == ["event_1001", "event_1002"]

    def test_list_events_empty_file(self, empty_hdf5_path: str) -> None:
        with self.loader.load_file(empty_hdf5_path) as f:
            events = self.loader.list_events(f)
        assert events == []

    def test_load_event_success(self, sample_hdf5_path: str) -> None:
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert event.event_id == "event_1001"
        assert event.uuid is not None
        assert event.timestamp > 0

    def test_load_event_not_found(self, sample_hdf5_path: str) -> None:
        with self.loader.load_file(sample_hdf5_path) as f:
            with pytest.raises(EventParseError, match="Event not found"):
                self.loader.load_event(f, "event_9999")

    def test_load_metadata(self, sample_hdf5_path: str) -> None:
        with self.loader.load_file(sample_hdf5_path) as f:
            metadata = self.loader.load_metadata(f)
        assert metadata.patient_id == "PT_TEST"
        assert metadata.sampling_rate_ecg == 200.0
        assert metadata.sampling_rate_ppg == 75.0
        assert metadata.sampling_rate_resp == 33.33
        assert metadata.alarm_offset_seconds == 6.0
        assert metadata.seconds_before_event == 6.0
        assert metadata.seconds_after_event == 6.0
        assert metadata.data_quality_score == 0.94
        assert metadata.device_info == "RMSAI-SimDevice-v1.0"
        assert metadata.max_vital_history == 30

    def test_event_has_metadata_attached(self, sample_hdf5_path: str) -> None:
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert event.metadata is not None
        assert event.metadata.patient_id == "PT_TEST"
