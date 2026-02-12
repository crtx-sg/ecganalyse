"""Preprocessing utilities: filtering and normalization for ECG signals."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Args:
        signal: 1-D signal array.
        fs: Sampling frequency in Hz.
        low: Low cutoff frequency in Hz.
        high: High cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal (same shape as input).
    """
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal).astype(signal.dtype)


def remove_baseline_wander(
    signal: np.ndarray,
    fs: float,
    cutoff: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Remove baseline wander using a high-pass filter.

    Args:
        signal: 1-D signal array.
        fs: Sampling frequency in Hz.
        cutoff: High-pass cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Signal with baseline wander removed.
    """
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, signal).astype(signal.dtype)


def normalize_leads(ecg_array: np.ndarray) -> np.ndarray:
    """Normalize each lead to zero-mean, unit-variance.

    Args:
        ecg_array: Array of shape [num_leads, num_samples].

    Returns:
        Normalized array of same shape. Leads with zero variance are left as
        zero-mean (no division).
    """
    mean = ecg_array.mean(axis=-1, keepdims=True)
    std = ecg_array.std(axis=-1, keepdims=True)
    # Avoid division by zero for constant signals
    std = np.where(std < 1e-10, 1.0, std)
    return ((ecg_array - mean) / std).astype(ecg_array.dtype)
