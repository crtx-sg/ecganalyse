"""ECG simulator facade — single entry point for generating synthetic events."""

from __future__ import annotations

import random as stdlib_random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.simulator.conditions import Condition, ConditionConfig, CONDITION_REGISTRY
from src.simulator.morphology import (
    PatientParams,
    generate_patient_params,
    generate_single_lead,
    _add_fibrillatory_waves,
    _add_flutter_waves,
    _add_vfib_chaos,
)
from src.simulator.noise import NoiseConfig, NOISE_PRESETS, apply_noise_pipeline


# Sampling rates matching Phase 0 loader contract
FS_ECG = 200.0
FS_PPG = 75.0
FS_RESP = 33.33
ECG_DURATION = 12.0

# Lead names matching the Phase 0 schema
LEAD_NAMES = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]


@dataclass
class SimulatedEvent:
    """Complete simulated alarm event."""

    condition: Condition
    hr: float
    ecg_signals: dict[str, np.ndarray]   # 7 leads
    ppg_signal: np.ndarray
    resp_signal: np.ndarray
    vitals: dict[str, dict]
    noise_level: str
    patient_params: PatientParams
    pacer_info: int = 0
    pacer_offset: int = 0


class ECGSimulator:
    """Facade for generating synthetic ECG data.

    Args:
        fs: ECG sampling frequency in Hz (default 200).
        duration: signal duration in seconds (default 12).
        seed: random seed for reproducibility. ``None`` for non-deterministic.
    """

    def __init__(
        self,
        fs: float = FS_ECG,
        duration: float = ECG_DURATION,
        seed: int | None = None,
    ) -> None:
        self.fs = fs
        self.duration = duration
        self.n_samples = int(fs * duration)
        self.time = np.linspace(0, duration, self.n_samples, endpoint=False)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_ecg(
        self,
        condition: Condition,
        hr: float | None = None,
        noise_level: str = "medium",
        noise_config: NoiseConfig | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate a 7-lead ECG for *condition*.

        Returns:
            Dictionary mapping lead name to 1-D float32 array of length ``n_samples``.
        """
        cfg = CONDITION_REGISTRY[condition]
        if hr is None:
            hr = self._rng.uniform(*cfg.hr_range)

        patient = generate_patient_params(self._rng)
        nc = noise_config or NOISE_PRESETS.get(noise_level, NOISE_PRESETS["medium"])

        # Generate Lead I and Lead II independently
        lead_I = self._generate_lead(condition, cfg, patient, hr, scale=1.0)
        lead_II = self._generate_lead(condition, cfg, patient, hr, scale=1.1)

        # Einthoven's law: III = II - I
        lead_III = lead_II - lead_I

        # Augmented limb leads derived from I and II
        aVR = -(lead_I + lead_II) / 2.0
        aVL = lead_I - lead_II / 2.0
        aVF = lead_II - lead_I / 2.0

        # Precordial-like lead (vVX) — independent generation
        vVX = self._generate_lead(condition, cfg, patient, hr, scale=1.2)

        raw = {
            "ECG1": lead_I,
            "ECG2": lead_II,
            "ECG3": lead_III,
            "aVR": aVR,
            "aVL": aVL,
            "aVF": aVF,
            "vVX": vVX,
        }

        # Apply noise to each lead independently
        signals: dict[str, np.ndarray] = {}
        for name, sig in raw.items():
            noisy = apply_noise_pipeline(sig, self.time, self.fs, self._rng, nc)
            signals[name] = noisy.astype(np.float32)

        return signals

    def generate_event(
        self,
        condition: Condition | None = None,
        hr: float | None = None,
        noise_level: str = "medium",
        noise_config: NoiseConfig | None = None,
        condition_proportions: dict[Condition, float] | None = None,
    ) -> SimulatedEvent:
        """Generate a complete alarm event (ECG + PPG + RESP + vitals).

        If *condition* is ``None`` a random condition is selected using
        *condition_proportions* (uniform over all conditions by default).
        """
        if condition is None:
            condition = self._pick_condition(condition_proportions)

        cfg = CONDITION_REGISTRY[condition]
        if hr is None:
            hr = float(self._rng.uniform(*cfg.hr_range))

        ecg_signals = self.generate_ecg(condition, hr, noise_level, noise_config)

        patient = generate_patient_params(self._rng)
        ppg = self._generate_ppg(hr, condition)
        resp = self._generate_resp(hr, condition)
        vitals = self._generate_vitals(hr, condition)
        pacer_info = self._generate_pacer_info(condition)
        pacer_offset = self._generate_pacer_offset()

        return SimulatedEvent(
            condition=condition,
            hr=hr,
            ecg_signals=ecg_signals,
            ppg_signal=ppg,
            resp_signal=resp,
            vitals=vitals,
            noise_level=noise_level,
            patient_params=patient,
            pacer_info=pacer_info,
            pacer_offset=pacer_offset,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_lead(
        self,
        condition: Condition,
        cfg: ConditionConfig,
        patient: PatientParams,
        hr: float,
        scale: float,
    ) -> np.ndarray:
        """Generate a single raw (noiseless) ECG lead."""
        sig = generate_single_lead(self.time, cfg, patient, hr, scale, self._rng)

        # Condition-specific post-processing that affects the whole waveform
        if condition == Condition.ATRIAL_FIBRILLATION:
            sig = _add_fibrillatory_waves(sig, self.time, self._rng)
        elif condition == Condition.ATRIAL_FLUTTER:
            sig = _add_flutter_waves(sig, self.time, self._rng)
        elif condition == Condition.VENTRICULAR_FIBRILLATION:
            sig = _add_vfib_chaos(sig, self.time, self._rng)

        return sig

    def _pick_condition(
        self, proportions: dict[Condition, float] | None,
    ) -> Condition:
        """Randomly pick a condition using optional proportions dict."""
        if proportions:
            conditions = list(proportions.keys())
            weights = [proportions[c] for c in conditions]
        else:
            conditions = list(Condition)
            weights = [1.0] * len(conditions)
        total = sum(weights)
        weights = [w / total for w in weights]
        idx = self._rng.choice(len(conditions), p=weights)
        return conditions[idx]

    # --- PPG ---

    def _generate_ppg(self, hr: float, condition: Condition) -> np.ndarray:
        n = int(ECG_DURATION * FS_PPG)
        t = np.linspace(0, ECG_DURATION, n, endpoint=False)
        freq = hr / 60.0
        ppg = np.sin(2 * np.pi * freq * t)
        ppg += 0.3 * np.sin(4 * np.pi * freq * t + np.pi / 4)

        if condition == Condition.ATRIAL_FIBRILLATION:
            ppg += 0.2 * self._rng.normal(0, 1, n)
        elif condition in (Condition.SINUS_TACHYCARDIA, Condition.SVT):
            ppg *= 0.8
        elif condition == Condition.SINUS_BRADYCARDIA:
            ppg *= 1.2

        noise = 0.05 * self._rng.normal(0, 1, n)
        baseline = 1.0 + 0.1 * np.sin(2 * np.pi * 0.1 * t)
        return ((ppg + noise + baseline) * 100).astype(np.float32)

    # --- Respiratory ---

    def _generate_resp(self, hr: float, condition: Condition) -> np.ndarray:
        n = int(ECG_DURATION * FS_RESP)
        t = np.linspace(0, ECG_DURATION, n, endpoint=False)

        if condition in (Condition.VENTRICULAR_TACHYCARDIA, Condition.VENTRICULAR_FIBRILLATION):
            resp_rate = self._rng.uniform(22, 30)
        elif condition == Condition.ATRIAL_FIBRILLATION:
            resp_rate = self._rng.uniform(18, 25)
        elif condition == Condition.SINUS_BRADYCARDIA:
            resp_rate = self._rng.uniform(12, 18)
        else:
            resp_rate = self._rng.uniform(12, 20)

        resp_freq = resp_rate / 60.0
        resp = np.sin(2 * np.pi * resp_freq * t)
        resp += 0.1 * np.sin(2 * np.pi * (hr / 60.0) * t)
        resp += 0.05 * np.sin(2 * np.pi * resp_freq * 0.1 * t)
        resp += 0.02 * self._rng.normal(0, 1, n)
        resp = resp * 1000 + self._rng.uniform(8000, 12000)
        return resp.astype(np.float32)

    # --- Vitals ---

    def _generate_vitals(self, hr: float, condition: Condition) -> dict[str, dict]:
        """Generate condition-dependent vital signs."""
        pulse = hr + self._rng.uniform(-2, 2)
        spo2 = self._rng.uniform(96, 99.5)
        temp_f = self._rng.uniform(36.6, 37.5) * 9 / 5 + 32
        resp_rate = self._rng.uniform(12, 20)

        if condition == Condition.NORMAL_SINUS:
            systolic = self._rng.uniform(110, 130)
            diastolic = self._rng.uniform(70, 85)
        elif condition in (Condition.SINUS_TACHYCARDIA, Condition.SVT):
            systolic = self._rng.uniform(130, 150)
            diastolic = self._rng.uniform(85, 95)
        elif condition == Condition.SINUS_BRADYCARDIA:
            systolic = self._rng.uniform(100, 120)
            diastolic = self._rng.uniform(60, 75)
        elif condition == Condition.ATRIAL_FIBRILLATION:
            systolic = self._rng.uniform(120, 160)
            diastolic = self._rng.uniform(80, 100)
        elif condition in (
            Condition.VENTRICULAR_TACHYCARDIA,
            Condition.VENTRICULAR_FIBRILLATION,
        ):
            systolic = self._rng.uniform(140, 180)
            diastolic = self._rng.uniform(90, 110)
            spo2 = self._rng.uniform(88, 95)
            resp_rate = self._rng.uniform(22, 30)
        else:
            systolic = self._rng.uniform(110, 140)
            diastolic = self._rng.uniform(70, 90)

        ts = time.time()
        return {
            "HR": {"value": int(round(hr)), "units": "bpm", "timestamp": ts,
                   "upper_threshold": 100, "lower_threshold": 60},
            "Pulse": {"value": int(round(pulse)), "units": "bpm", "timestamp": ts,
                      "upper_threshold": 100, "lower_threshold": 60},
            "SpO2": {"value": int(round(spo2)), "units": "%", "timestamp": ts,
                     "upper_threshold": 100, "lower_threshold": 90},
            "Systolic": {"value": int(round(systolic)), "units": "mmHg", "timestamp": ts,
                         "upper_threshold": 140, "lower_threshold": 90},
            "Diastolic": {"value": int(round(diastolic)), "units": "mmHg", "timestamp": ts,
                          "upper_threshold": 90, "lower_threshold": 60},
            "RespRate": {"value": int(round(resp_rate)), "units": "breaths/min", "timestamp": ts,
                         "upper_threshold": 20, "lower_threshold": 12},
            "Temp": {"value": round(temp_f, 1), "units": "\u00b0F", "timestamp": ts,
                     "upper_threshold": 100.4, "lower_threshold": 96.0},
            "XL_Posture": {"value": int(round(self._rng.uniform(-10, 45))),
                           "units": "degrees", "timestamp": ts,
                           "step_count": int(self._rng.integers(0, 5000)),
                           "time_since_posture_change": int(self._rng.integers(60, 3600))},
        }

    # --- Pacer ---

    def _generate_pacer_info(self, condition: Condition) -> int:
        if condition in (Condition.VENTRICULAR_TACHYCARDIA, Condition.VENTRICULAR_FIBRILLATION):
            pacer_type = self._rng.choice([0, 1, 2, 3], p=[0.6, 0.1, 0.2, 0.1])
        elif condition == Condition.SINUS_BRADYCARDIA:
            pacer_type = self._rng.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.4, 0.1])
        else:
            pacer_type = self._rng.choice([0, 1, 2, 3], p=[0.95, 0.02, 0.02, 0.01])
        if pacer_type == 0:
            return 0
        rate = int(self._rng.integers(60, 101))
        amp = int(self._rng.integers(1, 11))
        flags = int(self._rng.integers(0, 16))
        return int(
            (pacer_type & 0xFF)
            | ((rate & 0xFF) << 8)
            | ((amp & 0xFF) << 16)
            | ((flags & 0xFF) << 24)
        )

    def _generate_pacer_offset(self) -> int:
        max_samples = int(ECG_DURATION * FS_ECG)
        return int(self._rng.integers(int(max_samples * 0.2), int(max_samples * 0.8)))
