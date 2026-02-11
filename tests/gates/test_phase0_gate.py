"""Gate tests for Phase 0: Data Loading output contracts."""

import numpy as np
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader

pytestmark = pytest.mark.gate

EXPECTED_LEADS = {"ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"}


class TestPhase0OutputContract:
    """Validate that Phase 0 output matches the AlarmEvent contract."""

    def setup_method(self) -> None:
        self.loader = HDF5AlarmEventLoader()

    def test_ecg_shape_contract(self, sample_hdf5_path: str) -> None:
        """ECG must be [7, 2400] float32."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert event.ecg.as_array.shape == (7, 2400)
        assert event.ecg.as_array.dtype == np.float32

    def test_ecg_lead_names_contract(self, sample_hdf5_path: str) -> None:
        """ECG must have exactly 7 expected leads."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert set(event.ecg.leads) == EXPECTED_LEADS

    def test_ecg_sample_rate_contract(self, sample_hdf5_path: str) -> None:
        """ECG sample rate must be 200 Hz."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert event.ecg.sample_rate == 200

    def test_ecg_duration_contract(self, sample_hdf5_path: str) -> None:
        """ECG duration must be 12.0 seconds."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert event.ecg.duration_sec == 12.0

    def test_ecg_value_range_contract(self, sample_hdf5_path: str) -> None:
        """ECG signal values must be in physiological range [-10, 10] mV."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        for lead in event.ecg.leads:
            sig = event.ecg.signals[lead]
            assert sig.min() >= -10.0, f"{lead} min value {sig.min()} out of range"
            assert sig.max() <= 10.0, f"{lead} max value {sig.max()} out of range"

    def test_event_required_fields_contract(self, sample_hdf5_path: str) -> None:
        """AlarmEvent must have non-null event_id, uuid, timestamp."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert event.event_id is not None and event.event_id != ""
        assert event.uuid is not None and event.uuid != ""
        assert event.timestamp > 0

    def test_metadata_contract(self, sample_hdf5_path: str) -> None:
        """FileMetadata must have all required fields populated."""
        with self.loader.load_file(sample_hdf5_path) as f:
            metadata = self.loader.load_metadata(f)
        assert metadata.patient_id != ""
        assert metadata.sampling_rate_ecg == 200.0
        assert metadata.alarm_offset_seconds == 6.0
        assert metadata.seconds_before_event == 6.0
        assert metadata.seconds_after_event == 6.0
        assert 0.0 <= metadata.data_quality_score <= 1.0
        assert metadata.device_info != ""
        assert metadata.max_vital_history > 0

    def test_vitals_contract(self, sample_hdf5_path: str) -> None:
        """Vitals must have proper structure when present."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert event.vitals is not None
        vitals_dict = event.vitals.to_dict()
        # Must have required vital keys
        for key in ("hr", "spo2", "systolic", "diastolic"):
            assert key in vitals_dict, f"Missing vital: {key}"
            assert "value" in vitals_dict[key]
            assert "units" in vitals_dict[key]
            assert "timestamp" in vitals_dict[key]
            assert "threshold_violation" in vitals_dict[key]

    def test_ppg_shape_contract(self, sample_hdf5_path: str) -> None:
        """PPG signal must be [900] float32 when present."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert event.ppg is not None
        assert event.ppg.signal.shape == (900,)
        assert event.ppg.signal.dtype == np.float32
        assert event.ppg.sample_rate == 75.0

    def test_resp_shape_contract(self, sample_hdf5_path: str) -> None:
        """RESP signal must be [400] float32 when present."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert event.resp is not None
        assert event.resp.signal.shape == (400,)
        assert event.resp.signal.dtype == np.float32
        assert event.resp.sample_rate == 33.33

    def test_ecg_num_samples_contract(self, sample_hdf5_path: str) -> None:
        """ECG num_samples must match actual signal length."""
        with self.loader.load_file(sample_hdf5_path) as f:
            event = self.loader.load_event(f, "event_1001")
        assert event.ecg.num_samples == 2400
        for lead in event.ecg.leads:
            assert event.ecg.signals[lead].shape[0] == event.ecg.num_samples
