"""Integration tests for Phase 0: full HDF5 → AlarmEvent round-trip."""

import numpy as np
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.data.vitals_parser import VitalsParser

pytestmark = pytest.mark.integration


class TestPhase0Integration:
    """End-to-end integration tests for data loading phase."""

    def test_full_round_trip(self, sample_hdf5_path: str) -> None:
        """Load file → list events → load each event → verify structure."""
        loader = HDF5AlarmEventLoader()

        with loader.load_file(sample_hdf5_path) as f:
            metadata = loader.load_metadata(f)
            events = loader.list_events(f)

            assert len(events) == 2
            assert metadata.patient_id == "PT_TEST"

            for event_id in events:
                event = loader.load_event(f, event_id)

                # ECG present and valid
                assert event.ecg.as_array.shape == (7, 2400)
                assert event.ecg.sample_rate == 200

                # Event identity
                assert event.event_id == event_id
                assert event.uuid is not None
                assert event.timestamp > 0

                # Metadata attached
                assert event.metadata is not None
                assert event.metadata.patient_id == "PT_TEST"

    def test_vitals_threshold_integration(self, sample_hdf5_path: str) -> None:
        """Load event → extract vitals → check thresholds → no violations."""
        loader = HDF5AlarmEventLoader()
        vitals_parser = VitalsParser()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        assert event.vitals is not None
        violations = vitals_parser.check_threshold_violations(event.vitals)
        assert len(violations) == 0

    def test_ecg_signal_integrity(self, sample_hdf5_path: str) -> None:
        """Loaded ECG signals should have non-zero variance (not flat)."""
        loader = HDF5AlarmEventLoader()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        for lead in event.ecg.leads:
            sig = event.ecg.signals[lead]
            assert np.std(sig) > 0.001, f"Lead {lead} appears flat"

    def test_multiple_events_independent(self, sample_hdf5_path: str) -> None:
        """Each event should have independent data (not referencing same array)."""
        loader = HDF5AlarmEventLoader()

        with loader.load_file(sample_hdf5_path) as f:
            event1 = loader.load_event(f, "event_1001")
            event2 = loader.load_event(f, "event_1002")

        assert event1.event_id != event2.event_id
        assert event1.uuid != event2.uuid
        # Signals should be loaded independently
        assert event1.ecg.signals["ECG1"] is not event2.ecg.signals["ECG1"]

    def test_no_vitals_graceful(self, sample_hdf5_no_vitals: str) -> None:
        """Event without vitals group loads without error."""
        loader = HDF5AlarmEventLoader()

        with loader.load_file(sample_hdf5_no_vitals) as f:
            event = loader.load_event(f, "event_1001")

        assert event.ecg is not None
        assert event.ecg.as_array.shape == (7, 2400)
        assert event.vitals is None

    def test_no_auxiliary_graceful(self, sample_hdf5_no_aux: str) -> None:
        """Event without PPG/RESP loads without error."""
        loader = HDF5AlarmEventLoader()

        with loader.load_file(sample_hdf5_no_aux) as f:
            event = loader.load_event(f, "event_1001")

        assert event.ecg is not None
        assert event.ppg is None
        assert event.resp is None

    def test_vitals_to_dict_complete(self, sample_hdf5_path: str) -> None:
        """Vitals to_dict produces complete JSON-serializable output."""
        import json

        loader = HDF5AlarmEventLoader()

        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        assert event.vitals is not None
        d = event.vitals.to_dict()

        # Must be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed == d

        # Must have expected vitals
        assert "hr" in d
        assert "spo2" in d
        assert "posture" in d
