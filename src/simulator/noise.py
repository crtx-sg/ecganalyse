"""Composable noise pipeline for ECG signal corruption."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseConfig:
    """Configuration for the noise pipeline.

    Attributes:
        baseline_wander_amp: amplitude of baseline wander (mV).
        gaussian_std: standard deviation of additive Gaussian noise.
        emg_probability: probability of EMG artifact burst.
        motion_probability: probability of motion artifacts.
        powerline_probability: probability of 50/60 Hz interference.
        electrode_probability: probability of poor electrode contact scaling.
    """

    baseline_wander_amp: float = 0.10
    gaussian_std: float = 0.10
    emg_probability: float = 0.30
    motion_probability: float = 0.15
    powerline_probability: float = 0.20
    electrode_probability: float = 0.10


NOISE_PRESETS: dict[str, NoiseConfig] = {
    "clean": NoiseConfig(
        baseline_wander_amp=0.0,
        gaussian_std=0.0,
        emg_probability=0.0,
        motion_probability=0.0,
        powerline_probability=0.0,
        electrode_probability=0.0,
    ),
    "low": NoiseConfig(
        baseline_wander_amp=0.05,
        gaussian_std=0.05,
        emg_probability=0.10,
        motion_probability=0.05,
        powerline_probability=0.10,
        electrode_probability=0.05,
    ),
    "medium": NoiseConfig(
        baseline_wander_amp=0.10,
        gaussian_std=0.10,
        emg_probability=0.30,
        motion_probability=0.15,
        powerline_probability=0.20,
        electrode_probability=0.10,
    ),
    "high": NoiseConfig(
        baseline_wander_amp=0.15,
        gaussian_std=0.20,
        emg_probability=0.50,
        motion_probability=0.30,
        powerline_probability=0.30,
        electrode_probability=0.20,
    ),
}


def add_baseline_wander(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> np.ndarray:
    """Add low-frequency baseline wander (respiration, movement)."""
    if config.baseline_wander_amp == 0.0:
        return signal
    freq = rng.uniform(0.1, 0.5)
    amp = config.baseline_wander_amp
    wander = amp * np.sin(2 * np.pi * freq * time + rng.uniform(0, 2 * np.pi))
    wander += 0.5 * amp * np.sin(2 * np.pi * freq * 0.3 * time + rng.uniform(0, 2 * np.pi))
    return signal + wander


def add_gaussian_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> np.ndarray:
    """Add white Gaussian noise."""
    if config.gaussian_std == 0.0:
        return signal
    noise = rng.normal(0, config.gaussian_std, len(signal))
    return signal + noise


def add_emg_artifact(
    signal: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> np.ndarray:
    """Add high-frequency EMG (muscle) artifact burst."""
    if rng.random() >= config.emg_probability:
        return signal

    result = signal.copy()
    duration_samples = int(rng.uniform(0.5, 2.0) * fs)
    start = rng.integers(0, max(1, len(signal) - duration_samples))
    muscle_noise = rng.normal(0, 0.15, duration_samples)

    # Bandpass to EMG range (20–90 Hz) using simple windowed approach
    # Avoid scipy dependency by using a frequency-domain filter
    nyq = fs / 2.0
    if nyq > 20:
        freqs = np.fft.rfftfreq(duration_samples, d=1.0 / fs)
        spectrum = np.fft.rfft(muscle_noise)
        # Apply bandpass mask
        mask = (freqs >= 20) & (freqs <= min(90, nyq - 1))
        spectrum[~mask] = 0
        muscle_noise = np.fft.irfft(spectrum, n=duration_samples)

    end = min(start + duration_samples, len(result))
    result[start:end] += muscle_noise[: end - start]
    return result


def add_motion_artifact(
    signal: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> np.ndarray:
    """Add brief motion artifact spikes."""
    if rng.random() >= config.motion_probability:
        return signal

    result = signal.copy()
    n_spikes = rng.integers(1, 4)
    for _ in range(n_spikes):
        loc = rng.integers(100, max(101, len(signal) - 100))
        width = rng.integers(20, 100)
        amp = rng.uniform(0.2, 0.5) * rng.choice([-1, 1])
        spike = amp * np.exp(
            -((np.arange(width) - width / 2) ** 2) / (width / 6) ** 2
        )
        end = min(loc + width, len(result))
        result[loc:end] += spike[: end - loc]
    return result


def add_powerline_interference(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> np.ndarray:
    """Add 50/60 Hz powerline interference."""
    if rng.random() >= config.powerline_probability:
        return signal
    freq = rng.choice([50, 60])
    amp = rng.uniform(0.02, 0.08)
    return signal + amp * np.sin(2 * np.pi * freq * time + rng.uniform(0, 2 * np.pi))


def add_electrode_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> np.ndarray:
    """Simulate poor electrode contact via random amplitude scaling."""
    if rng.random() >= config.electrode_probability:
        return signal
    factor = rng.uniform(0.6, 0.9)
    return signal * factor


def apply_noise_pipeline(
    signal: np.ndarray,
    time: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> np.ndarray:
    """Apply the full noise pipeline in order."""
    out = add_baseline_wander(signal, time, rng, config)
    out = add_gaussian_noise(out, rng, config)
    out = add_emg_artifact(out, fs, rng, config)
    out = add_motion_artifact(out, rng, config)
    out = add_powerline_interference(out, time, rng, config)
    out = add_electrode_noise(out, rng, config)
    return out
