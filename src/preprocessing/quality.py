"""Signal quality assessment for ECG leads."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from config.settings import PreprocessingConfig
from src.ecg_system.schemas import ECGData, QualityReport


# Default lead order (matches ECGData.LEAD_ORDER)
_ALL_LEADS = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]


class SignalQualityAssessor:
    """Compute Signal Quality Index (SQI) per ECG lead and overall.

    SQI sub-metrics (each 0–1, combined as weighted average):
    - Variance check: penalises near-zero variance (lead-off).
    - Kurtosis: physiological ECG has kurtosis ~3–10; outside that range is penalised.
    - High-frequency noise ratio: fraction of power above 40 Hz.
    - Baseline stability: standard deviation of low-pass-filtered signal.
    """

    # Weights for combining sub-metrics into per-lead SQI
    _WEIGHTS = {
        "variance": 0.30,
        "kurtosis": 0.20,
        "hf_noise": 0.25,
        "baseline": 0.25,
    }

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, ecg: ECGData) -> QualityReport:
        """Assess quality of all leads in an ECGData object.

        Returns a QualityReport with per-lead SQI, usable/excluded leads,
        quality flags, noise level, and baseline stability.
        """
        fs = ecg.sample_rate
        lead_sqi: dict[str, float] = {}
        quality_flags: list[str] = []
        usable_leads: list[str] = []
        excluded_leads: list[str] = []

        for lead in _ALL_LEADS:
            signal = ecg.signals[lead]
            sqi = self.compute_sqi(signal, fs)
            lead_sqi[lead] = round(sqi, 4)

            if self.detect_lead_off(signal):
                quality_flags.append(f"lead_off_{lead}")
                excluded_leads.append(lead)
            elif self.detect_saturation(signal):
                quality_flags.append(f"saturation_{lead}")
                excluded_leads.append(lead)
            elif sqi < self.config.sqi_threshold:
                excluded_leads.append(lead)
            else:
                usable_leads.append(lead)

        # Pacer detection via extras field
        if self.check_pacer_presence(ecg.extras):
            quality_flags.append("pacer_detected")

        # Overall SQI: mean of usable leads, or mean of all if none usable
        if usable_leads:
            overall_sqi = float(np.mean([lead_sqi[l] for l in usable_leads]))
        else:
            overall_sqi = float(np.mean(list(lead_sqi.values())))
            quality_flags.append("signal_unusable")

        # Noise level classification
        noise_level = self._classify_noise(overall_sqi)

        # High-noise flag
        if noise_level == "high":
            quality_flags.append("high_noise")

        # Baseline stability (average across usable leads)
        baseline_stability = self._assess_baseline_stability(ecg, usable_leads or _ALL_LEADS)

        return QualityReport(
            overall_sqi=round(overall_sqi, 4),
            lead_sqi=lead_sqi,
            usable_leads=usable_leads,
            excluded_leads=excluded_leads,
            quality_flags=quality_flags,
            noise_level=noise_level,
            baseline_stability=baseline_stability,
        )

    def compute_sqi(self, signal: np.ndarray, fs: int) -> float:
        """Compute Signal Quality Index for a single lead (0.0–1.0)."""
        scores: dict[str, float] = {}

        # 1. Variance check
        var = float(np.var(signal))
        if var < 1e-6:
            # Near-zero variance → lead-off
            return 0.0
        # Sigmoid-like mapping: variance in physiological range (~0.01 to 1.0)
        scores["variance"] = min(1.0, var / 0.01) if var < 0.01 else 1.0

        # 2. Kurtosis
        kurt = float(_kurtosis(signal))
        # Physiological ECG: kurtosis 2–50 (sharp QRS complexes push it high)
        if 2.0 <= kurt <= 50.0:
            scores["kurtosis"] = 1.0
        elif kurt < 2.0:
            scores["kurtosis"] = max(0.0, kurt / 2.0)
        else:
            scores["kurtosis"] = max(0.0, 1.0 - (kurt - 50.0) / 50.0)

        # 3. High-frequency noise ratio
        hf_ratio = self._hf_power_ratio(signal, fs)
        scores["hf_noise"] = max(0.0, 1.0 - hf_ratio * 5.0)

        # 4. Baseline stability
        baseline_std = self._baseline_std(signal, fs)
        # Lower baseline wander std → better quality (scale: 0.5 → score 0.5)
        scores["baseline"] = max(0.0, 1.0 - baseline_std)

        # Weighted combination
        sqi = sum(self._WEIGHTS[k] * scores[k] for k in self._WEIGHTS)
        return float(np.clip(sqi, 0.0, 1.0))

    def detect_lead_off(self, signal: np.ndarray) -> bool:
        """Detect lead-off condition (flat line / near-zero variance)."""
        return float(np.var(signal)) < 1e-6

    def detect_saturation(self, signal: np.ndarray) -> bool:
        """Detect signal saturation (railing at extremes)."""
        # Saturation: >20% of samples at the same min or max value
        n = len(signal)
        max_val = float(np.max(signal))
        min_val = float(np.min(signal))
        at_max = np.sum(np.abs(signal - max_val) < 1e-6) / n
        at_min = np.sum(np.abs(signal - min_val) < 1e-6) / n
        return float(at_max) > 0.2 or float(at_min) > 0.2

    def check_pacer_presence(self, extras: dict) -> bool:
        """Check if pacemaker info is present in extras."""
        pacer_info = extras.get("pacer_info", 0)
        return pacer_info not in (0, None, "0")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_noise(overall_sqi: float) -> str:
        if overall_sqi >= 0.7:
            return "low"
        elif overall_sqi >= 0.4:
            return "moderate"
        return "high"

    def _hf_power_ratio(self, signal: np.ndarray, fs: int) -> float:
        """Fraction of signal power above 40 Hz."""
        fft_vals = np.fft.rfft(signal)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
        total_power = np.sum(power)
        if total_power < 1e-12:
            return 0.0
        hf_power = np.sum(power[freqs > 40.0])
        return float(hf_power / total_power)

    def _baseline_std(self, signal: np.ndarray, fs: int) -> float:
        """Standard deviation of the low-pass filtered baseline."""
        nyq = fs / 2.0
        cutoff = 0.5
        if cutoff / nyq >= 1.0:
            return 0.0
        sos = butter(2, cutoff / nyq, btype="lowpass", output="sos")
        baseline = sosfiltfilt(sos, signal)
        return float(np.std(baseline))

    def _assess_baseline_stability(
        self, ecg: ECGData, leads: list[str]
    ) -> str:
        """Classify baseline stability across selected leads."""
        stds = [
            self._baseline_std(ecg.signals[l], ecg.sample_rate)
            for l in leads
        ]
        mean_std = float(np.mean(stds))
        if mean_std < 0.1:
            return "stable"
        elif mean_std < 0.3:
            return "moderate"
        return "unstable"


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition)."""
    n = len(x)
    if n < 4:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    m4 = np.mean((x - mean) ** 4)
    return float(m4 / (std**4) - 3.0)
