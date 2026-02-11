"""End-to-end integration tests: generate -> write HDF5 -> load with Phase 0 loader."""

from pathlib import Path

import numpy as np
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.simulator.conditions import Condition
from src.simulator.ecg_simulator import ECGSimulator
from src.simulator.hdf5_writer import HDF5EventWriter

pytestmark = pytest.mark.integration


class TestSimulatorPhase0RoundTrip:
    """Verify that simulator-generated HDF5 files satisfy the Phase 0 loader contract."""

    @pytest.fixture
    def generated_path(self, tmp_path: Path) -> str:
        sim = ECGSimulator(seed=42)
        events = [
            sim.generate_event(Condition.NORMAL_SINUS, hr=72.0),
            sim.generate_event(Condition.ATRIAL_FIBRILLATION, hr=120.0),
            sim.generate_event(Condition.VENTRICULAR_TACHYCARDIA, hr=160.0),
        ]
        path = str(tmp_path / "sim_roundtrip.h5")
        HDF5EventWriter().write_file(path, events, patient_id="PT_SIM")
        return path

    def test_loader_opens_file(self, generated_path):
        loader = HDF5AlarmEventLoader()
        with loader.load_file(generated_path) as f:
            events = loader.list_events(f)
            assert len(events) == 3

    def test_metadata_loads(self, generated_path):
        loader = HDF5AlarmEventLoader()
        with loader.load_file(generated_path) as f:
            meta = loader.load_metadata(f)
            assert meta.patient_id == "PT_SIM"
            assert meta.sampling_rate_ecg == 200.0
            assert meta.sampling_rate_ppg == 75.0
            assert meta.max_vital_history == 30

    def test_ecg_shape_and_dtype(self, generated_path):
        loader = HDF5AlarmEventLoader()
        with loader.load_file(generated_path) as f:
            for event_id in loader.list_events(f):
                event = loader.load_event(f, event_id)
                arr = event.ecg.as_array
                assert arr.shape == (7, 2400), f"{event_id}: shape {arr.shape}"
                assert arr.dtype == np.float32

    def test_ecg_leads_correct(self, generated_path):
        loader = HDF5AlarmEventLoader()
        expected = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]
        with loader.load_file(generated_path) as f:
            event = loader.load_event(f, "event_1001")
            assert event.ecg.leads == expected

    def test_ecg_sample_rate(self, generated_path):
        loader = HDF5AlarmEventLoader()
        with loader.load_file(generated_path) as f:
            event = loader.load_event(f, "event_1001")
            assert event.ecg.sample_rate == 200

    def test_ecg_signals_non_flat(self, generated_path):
        loader = HDF5AlarmEventLoader()
        with loader.load_file(generated_path) as f:
            event = loader.load_event(f, "event_1001")
            for lead in event.ecg.leads:
                sig = event.ecg.signals[lead]
                assert np.std(sig) > 0.001, f"Lead {lead} appears flat"

    def test_ppg_loads(self, generated_path):
        loader = HDF5AlarmEventLoader()
        with loader.load_file(generated_path) as f:
            event = loader.load_event(f, "event_1001")
            assert event.ppg is not None
            assert event.ppg.signal.shape[0] == 900

    def test_resp_loads(self, generated_path):
        loader = HDF5AlarmEventLoader()
        with loader.load_file(generated_path) as f:
            event = loader.load_event(f, "event_1001")
            assert event.resp is not None

    def test_vitals_load(self, generated_path):
        loader = HDF5AlarmEventLoader()
        with loader.load_file(generated_path) as f:
            event = loader.load_event(f, "event_1001")
            assert event.vitals is not None
            assert event.vitals.hr is not None
            assert event.vitals.spo2 is not None
            assert event.vitals.posture is not None

    def test_event_identity(self, generated_path):
        loader = HDF5AlarmEventLoader()
        with loader.load_file(generated_path) as f:
            event = loader.load_event(f, "event_1001")
            assert event.event_id == "event_1001"
            assert event.uuid is not None
            assert event.timestamp > 0
