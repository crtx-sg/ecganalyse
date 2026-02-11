"""Beat morphology engine using Gaussian basis PQRST generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulator.conditions import ConditionConfig


@dataclass
class PatientParams:
    """Patient-specific physiological parameters for ECG generation."""

    pr_interval: float      # seconds (0.12–0.20)
    qrs_duration: float     # seconds (0.08–0.12 normal)
    qt_base: float          # seconds (0.35–0.44)
    p_wave_scale: float     # amplitude multiplier
    qrs_scale: float        # amplitude multiplier
    t_wave_scale: float     # amplitude multiplier
    p_wave_width: float     # seconds (0.08–0.12)
    qrs_sharpness: float    # 0.8–1.2
    t_wave_width: float     # seconds (0.12–0.20)
    baseline_drift_freq: float
    baseline_drift_amp: float


def generate_patient_params(rng: np.random.Generator) -> PatientParams:
    """Generate randomised patient-specific morphology parameters."""
    return PatientParams(
        pr_interval=rng.uniform(0.12, 0.20),
        qrs_duration=rng.uniform(0.08, 0.12),
        qt_base=rng.uniform(0.35, 0.44),
        p_wave_scale=rng.uniform(0.7, 1.3),
        qrs_scale=rng.uniform(0.8, 1.2),
        t_wave_scale=rng.uniform(0.7, 1.3),
        p_wave_width=rng.uniform(0.08, 0.12),
        qrs_sharpness=rng.uniform(0.8, 1.2),
        t_wave_width=rng.uniform(0.12, 0.20),
        baseline_drift_freq=rng.uniform(0.1, 0.5),
        baseline_drift_amp=rng.uniform(0.05, 0.15),
    )


def create_p_wave(
    time: np.ndarray,
    center: float,
    params: PatientParams,
    scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a P wave centred at *center* using Gaussian basis."""
    width = params.p_wave_width / 2.355  # FWHM → σ
    amplitude = 0.10 * params.p_wave_scale * scale
    asymmetry = rng.uniform(0.9, 1.1) if rng is not None else 1.0
    return amplitude * np.exp(-((time - center) ** 2) / (2 * (width * asymmetry) ** 2))


def create_qrs_complex(
    time: np.ndarray,
    center: float,
    params: PatientParams,
    scale: float = 1.0,
    wide: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create QRS complex with optional widening for ventricular conditions."""
    wave = np.zeros_like(time)
    qrs_dur = params.qrs_duration

    if wide:
        width_factor = rng.uniform(1.5, 2.0) if rng is not None else 1.75
        qrs_dur *= width_factor

    sharpness = params.qrs_sharpness
    amp = params.qrs_scale * scale

    _rng_scale = (lambda lo, hi: rng.uniform(lo, hi)) if rng is not None else (lambda lo, hi: (lo + hi) / 2)

    # Q wave
    q_center = center - qrs_dur * 0.15
    q_width = qrs_dur * 0.08 / 2.355
    q_amp = -0.10 * amp * _rng_scale(0.8, 1.2)
    wave += q_amp * np.exp(-((time - q_center) ** 2) / (2 * q_width ** 2))

    # R wave
    r_width = (qrs_dur * 0.35 / 2.355) / sharpness
    r_amp = 1.2 * amp * _rng_scale(0.9, 1.1)
    wave += r_amp * np.exp(-((time - center) ** 2) / (2 * r_width ** 2))

    # S wave
    s_center = center + qrs_dur * 0.15
    s_width = qrs_dur * 0.10 / 2.355
    s_amp = -0.25 * amp * _rng_scale(0.8, 1.2)
    wave += s_amp * np.exp(-((time - s_center) ** 2) / (2 * s_width ** 2))

    return wave


def create_t_wave(
    time: np.ndarray,
    center: float,
    params: PatientParams,
    scale: float = 1.0,
    inverted: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create T wave with optional inversion."""
    width = params.t_wave_width / 2.355
    amplitude = 0.25 * params.t_wave_scale * scale
    if inverted:
        amplitude *= -1
    asymmetry = rng.uniform(0.9, 1.1) if rng is not None else 1.0
    return amplitude * np.exp(-((time - center) ** 2) / (2 * (width * asymmetry) ** 2))


def add_st_elevation(
    signal: np.ndarray,
    beat_times: np.ndarray,
    fs: float,
    duration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add ST segment elevation localised to the ST segment of each beat."""
    elevated = signal.copy()
    st_start_offset = 0.08   # 80 ms after R peak
    st_duration = 0.12       # 120 ms ST segment

    for beat_time in beat_times:
        if beat_time + st_start_offset + st_duration >= duration:
            continue
        st_start_idx = int((beat_time + st_start_offset) * fs)
        st_end_idx = int((beat_time + st_start_offset + st_duration) * fs)
        st_elevation = rng.uniform(0.08, 0.15)
        seg_len = st_end_idx - st_start_idx
        if seg_len < 4:
            continue

        ramp_len = seg_len // 4
        plateau_len = seg_len // 2
        ramp_down_len = seg_len - ramp_len - plateau_len

        profile = np.concatenate([
            np.linspace(0, st_elevation, ramp_len),
            np.full(plateau_len, st_elevation),
            np.linspace(st_elevation, 0, ramp_down_len),
        ])
        end = min(st_start_idx + len(profile), len(elevated))
        elevated[st_start_idx:end] += profile[: end - st_start_idx]

    return elevated


def generate_beat_times(
    duration: float,
    hr: float,
    rr_irregularity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate beat onset times with optional RR irregularity.

    Args:
        duration: total signal duration in seconds.
        hr: heart rate in BPM.
        rr_irregularity: std-dev of jitter as fraction of mean RR interval.
        rng: numpy random generator.

    Returns:
        1-D array of beat times in seconds.
    """
    mean_rr = 60.0 / hr
    beat_times: list[float] = []
    t = 0.2  # start 200 ms in
    while t < duration - 0.5:
        beat_times.append(t)
        jitter = rng.normal(0, rr_irregularity * mean_rr)
        t += mean_rr + jitter
    return np.asarray(beat_times)


def _add_flutter_waves(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add sawtooth flutter (F) waves typical of atrial flutter."""
    flutter_rate = rng.uniform(250, 350) / 60.0  # ~4-6 Hz
    amplitude = rng.uniform(0.05, 0.12)
    phase = rng.uniform(0, 2 * np.pi)
    # Sawtooth approximation via harmonics
    f_wave = amplitude * np.sin(2 * np.pi * flutter_rate * time + phase)
    f_wave += 0.3 * amplitude * np.sin(4 * np.pi * flutter_rate * time + phase)
    return signal + f_wave


def _add_fibrillatory_waves(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add chaotic fibrillatory (f) waves for AF."""
    n_components = rng.integers(3, 7)
    f_waves = np.zeros_like(time)
    for _ in range(n_components):
        freq = rng.uniform(3, 8)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.02, 0.05)
        f_waves += amp * np.sin(2 * np.pi * freq * time + phase)
    return signal + f_waves


def _add_vfib_chaos(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create chaotic VFib waveform by summing many random sinusoids."""
    n_components = rng.integers(5, 12)
    chaos = np.zeros_like(time)
    for _ in range(n_components):
        freq = rng.uniform(2, 10)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.05, 0.25)
        chaos += amp * np.sin(2 * np.pi * freq * time + phase)
    return signal * 0.3 + chaos


def generate_single_lead(
    time: np.ndarray,
    condition_config: ConditionConfig,
    patient_params: PatientParams,
    hr: float,
    lead_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single-lead ECG waveform for the given condition.

    Args:
        time: time array (seconds).
        condition_config: morphology config for the condition.
        patient_params: patient-specific parameters.
        hr: heart rate in BPM.
        lead_scale: amplitude scaling for this lead.
        rng: numpy random generator.

    Returns:
        1-D float64 signal array.
    """
    duration = time[-1] + (time[1] - time[0])
    fs = 1.0 / (time[1] - time[0])

    beat_times = generate_beat_times(
        duration, hr, condition_config.rr_irregularity, rng,
    )

    signal = np.zeros_like(time)

    for bt in beat_times:
        # P wave
        if rng.random() < condition_config.p_wave_presence:
            pr = patient_params.pr_interval
            # First-degree AV block: prolong PR interval
            if condition_config.hr_range[0] >= 55 and condition_config.p_wave_presence > 0.9:
                if condition_config.qrs_duration_range == (0.08, 0.12) and condition_config.rr_irregularity == 0.03:
                    # Check if this is AV_BLOCK_1 by HR range lower bound
                    pass  # Use default PR
            p_center = bt - pr
            if p_center > 0:
                signal += create_p_wave(time, p_center, patient_params, lead_scale, rng)

        # QRS complex
        signal += create_qrs_complex(
            time, bt, patient_params, lead_scale,
            wide=condition_config.wide_qrs, rng=rng,
        )

        # T wave
        beat_interval = 60.0 / hr
        t_center = bt + patient_params.qt_base * np.sqrt(beat_interval)
        inverted = rng.random() < condition_config.t_wave_inversion_prob
        signal += create_t_wave(
            time, t_center, patient_params, lead_scale,
            inverted=inverted, rng=rng,
        )

    # Condition-specific post-processing
    if condition_config.st_elevation:
        signal = add_st_elevation(signal, beat_times, fs, duration, rng)

    return signal
