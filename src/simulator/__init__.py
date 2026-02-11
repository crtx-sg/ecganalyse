"""ECG Simulator — synthetic MIT-BIH-like ECG data generation."""

from src.simulator.conditions import Condition, ConditionConfig, CONDITION_REGISTRY
from src.simulator.morphology import PatientParams, generate_patient_params
from src.simulator.noise import NoiseConfig, NOISE_PRESETS
from src.simulator.ecg_simulator import ECGSimulator, SimulatedEvent
from src.simulator.hdf5_writer import HDF5EventWriter

__all__ = [
    "Condition",
    "ConditionConfig",
    "CONDITION_REGISTRY",
    "PatientParams",
    "generate_patient_params",
    "NoiseConfig",
    "NOISE_PRESETS",
    "ECGSimulator",
    "SimulatedEvent",
    "HDF5EventWriter",
]
