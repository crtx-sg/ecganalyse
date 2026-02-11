#!/usr/bin/env python3
"""
RMSAI HDF5 Data Generator with Configurable Anomaly Proportions
===============================================================

Generates synthetic ECG datasets in HDF5 format with configurable proportions of different
cardiac conditions for training and testing anomaly detection systems.

Features:
- Configurable anomaly/normal event proportions
- 5 cardiac conditions: Normal, Tachycardia, Bradycardia, Atrial Fibrillation, Ventricular Tachycardia
- Multi-modal signals: ECG (7 leads), PPG, respiratory, vital signs
- Realistic cardiac morphology for each condition
- Flexible command-line interface with preset configurations

Usage Examples:
  python rmsai_sim_hdf5_data.py 20 --normal 50              # 50% normal, 50% abnormal events
  python rmsai_sim_hdf5_data.py 15 --all-abnormal           # Only abnormal events
  python rmsai_sim_hdf5_data.py 30 --balanced               # Equal distribution across all conditions
  python rmsai_sim_hdf5_data.py 25 --high-anomaly           # 5% normal, 95% abnormal
  python rmsai_sim_hdf5_data.py 10 --proportions 0.1 0.2 0.2 0.25 0.25  # Custom exact proportions

Author: RMSAI Team
"""

import h5py
import numpy as np
import random
import argparse
from datetime import datetime, timedelta
import time
import uuid
import os
import json

# --- Configuration Constants ---
FS_ECG = 200.0          # ECG sampling frequency in Hz
FS_PPG = 75.0           # PPG sampling frequency in Hz
FS_RESP = 33.33         # Respiratory waveform sampling frequency in Hz (was impedance)
FS_VITALS = 8.33        # Vitals sampling frequency in Hz
ECG_DURATION = 12       # Duration of ECG signal in seconds
DATA_PACKET_INTERVAL = 0.12  # Data packet interval in seconds

def generate_patient_id():
    """Generate a realistic patient ID."""
    return f"PT{random.randint(1000, 9999)}"

def generate_synthetic_beat(hr, fs=FS_ECG, qrs_dur=0.04, condition='Normal'):
    """
    Generates a single, synthetic ECG heartbeat (PQRST complex) waveform
    tailored for specific cardiac conditions.
    """
    beat_duration_sec = 60.0 / hr
    num_samples_beat = int(beat_duration_sec * fs)
    t = np.linspace(0, beat_duration_sec, num_samples_beat, endpoint=False)

    beat = np.zeros(num_samples_beat)

    # Modify waveform characteristics based on condition
    if 'Ventricular Tachycardia' in condition:
        # Wide QRS complex, abnormal morphology
        qrs_dur = 0.12
        p_amp = 0.05  # Diminished P-wave
        r_amp = 1.2   # Enlarged R-wave
        t_amp = 0.15  # Reduced T-wave
    elif 'Atrial Fibrillation' in condition:
        # Irregular rhythm, no clear P-wave
        p_amp = 0.02  # Virtually absent P-wave
        r_amp = 1.0
        t_amp = 0.22
        # Add irregular baseline
        beat += 0.03 * np.random.normal(0, 1, num_samples_beat)
    elif condition == 'Tachycardia':
        # Fast rate, normal morphology
        p_amp = 0.08
        r_amp = 1.0
        t_amp = 0.18
    elif condition == 'Bradycardia':
        # Slow rate, may have enhanced P-wave
        p_amp = 0.12
        r_amp = 1.0
        t_amp = 0.25
    else:  # Normal
        p_amp = 0.1
        r_amp = 1.0
        t_amp = 0.22

    # P-wave
    p_start, p_dur = 0.10, 0.09
    if p_start + p_dur < beat_duration_sec:
        beat += p_amp * np.sin(np.pi * (t - p_start) / p_dur)**2 * \
                ((t >= p_start) & (t <= p_start + p_dur))

    # QRS complex
    q_start, q_dur, q_amp = 0.20, 0.01, -0.15
    if q_start + q_dur < beat_duration_sec:
        beat += q_amp * np.sin(np.pi * (t - q_start) / q_dur)**2 * \
                ((t >= q_start) & (t <= q_start + q_dur))

    r_start = 0.21
    if r_start + qrs_dur < beat_duration_sec:
        beat += r_amp * np.sin(np.pi * (t - r_start) / qrs_dur)**2 * \
                ((t >= r_start) & (t <= r_start + qrs_dur))

    s_start, s_dur, s_amp = 0.25, 0.02, -0.3
    if s_start + s_dur < beat_duration_sec:
        beat += s_amp * np.sin(np.pi * (t - s_start) / s_dur)**2 * \
                ((t >= s_start) & (t <= s_start + s_dur))

    # T-wave
    t_start, t_dur = 0.40, 0.18
    if t_start + t_dur < beat_duration_sec:
        beat += t_amp * np.sin(np.pi * (t - t_start) / t_dur)**2 * \
                ((t >= t_start) & (t <= t_start + t_dur))

    return beat

def generate_pacer_info(condition):
    """Generate pacer information as 4-byte integer."""
    # Pacer information encoded as bit flags:
    # Bit 0-7: Pacer type (0=None, 1=Single, 2=Dual, 3=Biventricular)
    # Bit 8-15: Pacer rate (if applicable)
    # Bit 16-23: Pacer amplitude (arbitrary units)
    # Bit 24-31: Status flags

    if 'Ventricular Tachycardia' in condition:
        # Higher chance of having a pacer
        pacer_type = random.choices([0, 1, 2, 3], weights=[0.6, 0.1, 0.2, 0.1])[0]
    elif condition == 'Bradycardia':
        # Very high chance of having a pacer
        pacer_type = random.choices([0, 1, 2, 3], weights=[0.2, 0.3, 0.4, 0.1])[0]
    else:
        # Low chance of having a pacer
        pacer_type = random.choices([0, 1, 2, 3], weights=[0.95, 0.02, 0.02, 0.01])[0]

    if pacer_type == 0:
        return 0  # No pacer

    # Generate pacer parameters
    pacer_rate = random.randint(60, 100) if pacer_type > 0 else 0
    pacer_amplitude = random.randint(1, 10) if pacer_type > 0 else 0
    status_flags = random.randint(0, 15) if pacer_type > 0 else 0

    # Pack into 32-bit integer
    pacer_info = (pacer_type & 0xFF) | \
                 ((pacer_rate & 0xFF) << 8) | \
                 ((pacer_amplitude & 0xFF) << 16) | \
                 ((status_flags & 0xFF) << 24)

    return int(pacer_info)

def generate_pacer_offset(condition):
    """Generate pacer offset as integer (samples from start of ECG window)."""
    # Pacer offset represents the sample number where pacer spike occurs
    # in the ECG window (0 to ECG_DURATION * FS_ECG samples)

    max_samples = int(ECG_DURATION * FS_ECG)  # 2400 samples for 12 seconds at 200 Hz

    if 'Ventricular Tachycardia' in condition or condition == 'Bradycardia':
        # For arrhythmias, pacer might be more strategically timed
        # Place offset in first or last quarter of the window
        if random.random() < 0.5:
            # Early pacing
            offset = random.randint(int(max_samples * 0.1), int(max_samples * 0.25))
        else:
            # Late pacing
            offset = random.randint(int(max_samples * 0.75), int(max_samples * 0.9))
    else:
        # For normal conditions, pacer can occur anywhere in the window
        offset = random.randint(int(max_samples * 0.2), int(max_samples * 0.8))

    return int(offset)

def generate_respiratory_waveform(hr, condition, duration=ECG_DURATION, fs=FS_RESP):
    """Generate respiratory waveform signal."""
    num_samples_total = int(duration * fs)

    # Base respiratory rate based on condition
    if 'Ventricular Tachycardia' in condition:
        resp_rate = random.uniform(22, 30)  # Elevated due to distress
    elif 'Atrial Fibrillation' in condition:
        resp_rate = random.uniform(18, 25)  # Slightly elevated
    elif condition == 'Bradycardia':
        resp_rate = random.uniform(12, 18)  # Normal to slightly low
    else:
        resp_rate = random.uniform(12, 20)  # Normal range

    # Generate time array
    t = np.linspace(0, duration, num_samples_total, endpoint=False)

    # Base respiratory waveform (sinusoidal)
    resp_freq = resp_rate / 60.0  # Convert to Hz
    respiratory = np.sin(2 * np.pi * resp_freq * t)

    # Add cardiac influence (slight modulation from heart rate)
    cardiac_freq = hr / 60.0
    cardiac_influence = 0.1 * np.sin(2 * np.pi * cardiac_freq * t)
    respiratory += cardiac_influence

    # Add breathing variability and noise
    respiratory += 0.05 * np.sin(2 * np.pi * resp_freq * 0.1 * t)  # Low frequency variation
    respiratory += 0.02 * np.random.normal(0, 1, num_samples_total)  # Noise

    # Scale to realistic amplitude range
    respiratory = respiratory * 1000 + random.uniform(8000, 12000)  # Offset around 10000

    return respiratory.astype(np.float32)

def generate_ecg_lead(hr, condition, duration=ECG_DURATION, fs=FS_ECG, lead_type='I'):
    """Generate ECG waveform for a specific lead."""
    num_samples_total = int(duration * fs)
    ecg_wave = np.array([])

    # Generate beats until duration is filled
    while len(ecg_wave) < num_samples_total:
        single_beat = generate_synthetic_beat(hr, fs, condition=condition)

        # Add variability for Atrial Fibrillation
        if 'Atrial Fibrillation' in condition:
            variation = random.uniform(-0.15, 0.15)
            beat_samples = int(len(single_beat) * (1 + variation))
            single_beat = np.resize(single_beat, beat_samples)

        ecg_wave = np.append(ecg_wave, single_beat)

    ecg_wave = ecg_wave[:num_samples_total]

    # Apply lead-specific scaling and characteristics
    lead_multipliers = {
        'I': 1.0,
        'II': 1.1,
        'III': 0.0,  # Will be calculated as II - I
        'aVR': -0.5,
        'aVL': 0.5,
        'aVF': 0.8,
        'vVX': 1.2
    }

    if lead_type != 'III':
        ecg_wave *= lead_multipliers.get(lead_type, 1.0)

    # Add realistic noise
    baseline_freq = 0.1
    baseline_amp = 0.05
    t_total = np.linspace(0, duration, num_samples_total, endpoint=False)
    baseline_wander = baseline_amp * np.sin(2 * np.pi * baseline_freq * t_total +
                                          random.uniform(0, 2*np.pi))

    noise_amp = 0.02
    hf_noise = noise_amp * np.random.normal(0, 1, num_samples_total)

    return ecg_wave + baseline_wander + hf_noise

def generate_ppg_signal(hr, condition, duration=ECG_DURATION, fs=FS_PPG):
    """Generate PPG (photoplethysmogram) signal."""
    num_samples_total = int(duration * fs)
    t = np.linspace(0, duration, num_samples_total, endpoint=False)

    # Base PPG waveform (simplified)
    beat_freq = hr / 60.0
    ppg_wave = np.sin(2 * np.pi * beat_freq * t)

    # Add systolic peak characteristic of PPG
    systolic_component = 0.3 * np.sin(4 * np.pi * beat_freq * t + np.pi/4)
    ppg_wave += systolic_component

    # Condition-specific modifications
    if 'Atrial Fibrillation' in condition:
        # Irregular amplitude variations
        irregularity = 0.2 * np.random.normal(0, 1, num_samples_total)
        ppg_wave += irregularity
    elif condition == 'Tachycardia':
        ppg_wave *= 0.8  # Reduced amplitude due to fast rate
    elif condition == 'Bradycardia':
        ppg_wave *= 1.2  # Enhanced amplitude due to slow rate

    # Add noise and baseline
    noise = 0.05 * np.random.normal(0, 1, num_samples_total)
    baseline = 1.0 + 0.1 * np.sin(2 * np.pi * 0.1 * t)

    return (ppg_wave + noise + baseline) * 100  # Convert to mV scale

def generate_impedance_signals(hr, condition, duration=ECG_DURATION, fs=FS_RESP):
    """Generate thoracic impedance and respiration signals."""
    num_samples_total = int(duration * fs)
    t = np.linspace(0, duration, num_samples_total, endpoint=False)

    # Respiration rate (12-20 breaths per minute)
    resp_rate = random.uniform(12, 20)
    resp_freq = resp_rate / 60.0

    # Base impedance signal
    impedance_baseline = 50  # Base impedance in Ohms
    resp_component = 5 * np.sin(2 * np.pi * resp_freq * t)  # Breathing component
    cardiac_component = 0.5 * np.sin(2 * np.pi * hr/60 * t)  # Cardiac component

    impedance = impedance_baseline + resp_component + cardiac_component

    # Respiration signal in breaths per minute
    respiration = np.full(num_samples_total, resp_rate)

    # Add condition-specific variations
    if condition == 'Tachycardia':
        respiration += random.uniform(2, 5)  # Increased respiration
    elif 'Ventricular Tachycardia' in condition:
        respiration += random.uniform(5, 8)  # Significantly increased
        impedance += 2 * np.random.normal(0, 1, num_samples_total)  # More noise

    # Add noise
    impedance += 0.5 * np.random.normal(0, 1, num_samples_total)
    respiration += 0.5 * np.random.normal(0, 1, num_samples_total)

    return impedance, respiration

def generate_vital_history(vital_name, current_value, current_timestamp, condition, num_samples=30):
    """Generate historical vital sign data for the specified number of past samples."""
    history = []

    # Generate history going backwards in time from current timestamp
    # Intervals between measurements vary by vital type (realistic medical intervals)
    interval_ranges = {
        'HR': (60, 300),           # 1-5 minutes between HR measurements
        'Pulse': (60, 300),        # 1-5 minutes between pulse measurements
        'SpO2': (30, 180),         # 30 seconds to 3 minutes for SpO2
        'Systolic': (300, 1800),   # 5-30 minutes between BP measurements
        'Diastolic': (300, 1800),  # 5-30 minutes between BP measurements
        'RespRate': (120, 600),    # 2-10 minutes between respiratory rate
        'Temp': (1800, 3600),     # 30 minutes to 1 hour between temperature
        'XL_Posture': (10, 60)     # 10 seconds to 1 minute for posture
    }

    min_interval, max_interval = interval_ranges.get(vital_name, (60, 300))

    # Generate baseline variation patterns based on condition and vital type
    if vital_name == 'HR':
        if condition == 'Tachycardia':
            baseline_trend = random.uniform(-5, 2)  # Slight downward trend as condition develops
            variation = 8
        elif condition == 'Bradycardia':
            baseline_trend = random.uniform(-2, 5)  # Slight upward trend
            variation = 5
        elif 'Atrial Fibrillation' in condition:
            baseline_trend = 0
            variation = 15  # More irregular
        else:
            baseline_trend = 0
            variation = 5
    elif vital_name in ['Systolic', 'Diastolic']:
        # Blood pressure trends
        baseline_trend = random.uniform(-1, 1)
        variation = 8 if condition != 'Normal' else 5
    elif vital_name == 'SpO2':
        if 'Ventricular Tachycardia' in condition:
            baseline_trend = random.uniform(-0.5, 1.5)  # Recovering SpO2
            variation = 3
        else:
            baseline_trend = 0
            variation = 1
    elif vital_name == 'Temp':
        baseline_trend = random.uniform(-0.1, 0.1)  # Very stable
        variation = 1.0
    else:
        baseline_trend = 0
        variation = random.uniform(1, 3)

    current_time = current_timestamp
    current_val = current_value

    # Generate historical samples going backwards
    for i in range(num_samples):
        # Calculate time interval (seconds ago)
        interval = random.uniform(min_interval, max_interval)
        historical_timestamp = current_time - interval * (i + 1)

        # Calculate historical value with trend and variation
        trend_offset = baseline_trend * (i + 1) * 0.1  # Gradual trend over time
        random_variation = random.uniform(-variation, variation)
        historical_value = current_val + trend_offset + random_variation

        # Apply realistic bounds for each vital type
        if vital_name == 'HR':
            historical_value = max(30, min(220, historical_value))
        elif vital_name == 'Pulse':
            historical_value = max(30, min(220, historical_value))
        elif vital_name == 'SpO2':
            historical_value = max(70, min(100, historical_value))
        elif vital_name == 'Systolic':
            historical_value = max(70, min(250, historical_value))
        elif vital_name == 'Diastolic':
            historical_value = max(40, min(150, historical_value))
        elif vital_name == 'RespRate':
            historical_value = max(8, min(40, historical_value))
        elif vital_name == 'Temp':
            historical_value = max(94.0, min(108.0, historical_value))  # Fahrenheit
        elif vital_name == 'XL_Posture':
            historical_value = max(-90, min(90, historical_value))

        # Round appropriately for vital type
        if vital_name in ['HR', 'Pulse', 'SpO2', 'Systolic', 'Diastolic', 'RespRate', 'XL_Posture']:
            historical_value = int(round(historical_value))
        else:  # Temperature
            historical_value = round(historical_value, 1)

        history.append({
            "value": historical_value,
            "timestamp": historical_timestamp
        })

    # Sort by timestamp (oldest first)
    history.sort(key=lambda x: x["timestamp"])

    return history

def generate_vitals_single(hr, condition, event_timestamp):
    """Generate single vital sign values with individual timestamps and thresholds."""
    # Generate base vitals
    pulse_rate = hr + random.uniform(-2, 2)  # PPG-derived pulse rate
    spo2_base = random.uniform(96, 99.5)
    temp_base = random.uniform(36.6, 37.5) * 9/5 + 32  # Convert to Fahrenheit
    resp_rate_base = random.uniform(12, 20)  # Normal respiratory rate range

    # Blood pressure based on condition
    if condition == 'Normal':
        systolic = random.uniform(110, 130)
        diastolic = random.uniform(70, 85)
    elif condition == 'Tachycardia':
        systolic = random.uniform(130, 150)
        diastolic = random.uniform(85, 95)
    elif condition == 'Bradycardia':
        systolic = random.uniform(100, 120)
        diastolic = random.uniform(60, 75)
    elif 'Atrial Fibrillation' in condition:
        systolic = random.uniform(120, 160)
        diastolic = random.uniform(80, 100)
    else:  # Ventricular Tachycardia
        systolic = random.uniform(140, 180)
        diastolic = random.uniform(90, 110)
        spo2_base = random.uniform(88, 95)  # Lower SpO2 due to poor perfusion
        resp_rate_base = random.uniform(22, 30)  # Elevated respiratory rate due to distress

    # Posture angle and activity data
    posture_base = random.uniform(-10, 45)  # degrees
    step_count = random.randint(0, 5000)  # Steps since last measurement
    time_since_posture_change = random.randint(60, 3600)  # 1 minute to 1 hour

    # Generate epoch timestamp for event
    event_epoch = time.mktime(event_timestamp.timetuple()) + event_timestamp.microsecond / 1e6

    # Generate individual timestamps for vitals (some may be spot measurements)
    # Vitals can be measured at different times around the event
    base_time_offsets = {
        'HR': random.uniform(-5, 5),        # Within 5 seconds of event
        'Pulse': random.uniform(-3, 3),     # Within 3 seconds of event
        'SpO2': random.uniform(-2, 2),      # Within 2 seconds of event
        'Systolic': random.uniform(-10, 10), # BP can be measured further from event
        'Diastolic': random.uniform(-10, 10),
        'RespRate': random.uniform(-3, 3),  # Respiratory rate within 3 seconds
        'Temp': random.uniform(-30, 30),    # Temperature less time-critical
        'XL_Posture': random.uniform(-1, 1) # Posture near event time
    }

    return {
        'HR': {
            'value': int(round(hr)),
            'units': 'bpm',
            'timestamp': event_epoch + base_time_offsets['HR'],
            'upper_threshold': 100,  # Standard HR upper limit
            'lower_threshold': 60    # Standard HR lower limit
        },
        'Pulse': {
            'value': int(round(pulse_rate)),
            'units': 'bpm',
            'timestamp': event_epoch + base_time_offsets['Pulse'],
            'upper_threshold': 100,  # Standard pulse upper limit
            'lower_threshold': 60    # Standard pulse lower limit
        },
        'SpO2': {
            'value': int(round(spo2_base)),
            'units': '%',
            'timestamp': event_epoch + base_time_offsets['SpO2'],
            'upper_threshold': 100,  # SpO2 upper limit (100%)
            'lower_threshold': 90    # SpO2 lower critical limit
        },
        'Systolic': {
            'value': int(round(systolic)),
            'units': 'mmHg',
            'timestamp': event_epoch + base_time_offsets['Systolic'],
            'upper_threshold': 140,  # Hypertension threshold
            'lower_threshold': 90    # Hypotension threshold
        },
        'Diastolic': {
            'value': int(round(diastolic)),
            'units': 'mmHg',
            'timestamp': event_epoch + base_time_offsets['Diastolic'],
            'upper_threshold': 90,   # Diastolic hypertension threshold
            'lower_threshold': 60    # Diastolic hypotension threshold
        },
        'RespRate': {
            'value': int(round(resp_rate_base)),
            'units': 'breaths/min',
            'timestamp': event_epoch + base_time_offsets['RespRate'],
            'upper_threshold': 20,   # Tachypnea threshold
            'lower_threshold': 12    # Bradypnea threshold
        },
        'Temp': {
            'value': round(temp_base, 1),
            'units': '°F',
            'timestamp': event_epoch + base_time_offsets['Temp'],
            'upper_threshold': 100.4,  # Fever threshold (°F)
            'lower_threshold': 96.0    # Hypothermia threshold (°F)
        },
        'XL_Posture': {
            'value': int(round(posture_base)),
            'units': 'degrees',
            'timestamp': event_epoch + base_time_offsets['XL_Posture'],
            'step_count': step_count,  # Steps since last measurement
            'time_since_posture_change': time_since_posture_change  # Seconds since last posture change
        }
    }

def generate_timestamps(start_time, duration, fs):
    """Generate timestamp arrays for different sampling rates."""
    num_samples = int(duration * fs)
    timestamps = []

    for i in range(num_samples):
        sample_time = start_time + timedelta(seconds=i/fs)
        timestamps.append(sample_time.isoformat())

    return timestamps

def generate_event_timestamps(num_events, start_time=None):
    """Generate timestamps for alarm events."""
    if start_time is None:
        start_time = datetime.now() - timedelta(hours=random.uniform(1, 24))

    timestamps = []
    current_time = start_time

    for i in range(num_events):
        timestamps.append(current_time)
        # Add realistic interval between events (30 seconds to 5 minutes)
        interval_seconds = random.uniform(30, 300)
        current_time += timedelta(seconds=interval_seconds)

    return timestamps

def validate_and_normalize_proportions(proportions):
    """Validate and normalize condition proportions."""
    if proportions is None:
        # Default proportions: 10% normal, 90% abnormal (distributed among 4 abnormal conditions)
        return [0.1, 0.225, 0.225, 0.225, 0.225]

    if len(proportions) != 5:
        raise ValueError(f"Expected 5 proportions for conditions, got {len(proportions)}")

    if any(p < 0 for p in proportions):
        raise ValueError("All proportions must be non-negative")

    total = sum(proportions)
    if total == 0:
        raise ValueError("At least one proportion must be greater than 0")

    # Normalize to sum to 1.0
    return [p / total for p in proportions]

def generate_condition_and_hr(condition_proportions=None):
    """Generate a condition and corresponding heart rate with configurable proportions."""
    conditions = [
        'Normal',
        'Tachycardia',
        'Bradycardia',
        'Atrial Fibrillation (PTB-XL)',
        'Ventricular Tachycardia (MIT-BIH)'
    ]

    # Use provided proportions or default
    condition_weights = validate_and_normalize_proportions(condition_proportions)

    condition = random.choices(conditions, condition_weights)[0]

    # Generate heart rate based on condition
    if condition == 'Normal':
        hr = round(random.uniform(65, 95), 1)
    elif condition == 'Tachycardia':
        hr = round(random.uniform(105, 140), 1)
    elif condition == 'Bradycardia':
        hr = round(random.uniform(40, 55), 1)
    elif 'Atrial Fibrillation' in condition:
        hr = round(random.uniform(110, 160), 1)
    elif 'Ventricular Tachycardia' in condition:
        hr = round(random.uniform(150, 190), 1)

    return condition, hr

def create_metadata_group(hf, patient_id, event_timestamps):
    """Create the metadata group with all required fields."""
    metadata_group = hf.create_group('metadata')

    # Patient identification (first member)
    metadata_group.create_dataset('patient_id', data=np.bytes_(patient_id))

    # Sampling rates
    metadata_group.create_dataset('sampling_rate_ecg', data=FS_ECG)
    metadata_group.create_dataset('sampling_rate_ppg', data=FS_PPG)
    metadata_group.create_dataset('sampling_rate_resp', data=FS_RESP)

    # Alarm timing (use first event as reference)
    alarm_time = event_timestamps[0]
    alarm_epoch = time.mktime(alarm_time.timetuple()) + alarm_time.microsecond / 1e6
    metadata_group.create_dataset('alarm_time_epoch', data=alarm_epoch)
    metadata_group.create_dataset('alarm_offset_seconds', data=ECG_DURATION / 2)  # Alarm at center of 12-sec window

    # Event timing configuration
    metadata_group.create_dataset('seconds_before_event', data=ECG_DURATION / 2)  # 6 seconds before alarm
    metadata_group.create_dataset('seconds_after_event', data=ECG_DURATION / 2)   # 6 seconds after alarm

    # Data quality score (random but realistic)
    metadata_group.create_dataset('data_quality_score', data=random.uniform(0.85, 0.98))

    # Device information
    metadata_group.create_dataset('device_info', data=np.bytes_('RMSAI-SimDevice-v1.0'))

    # Vitals history configuration
    metadata_group.create_dataset('max_vital_history', data=30)  # Default: store last 30 samples

    return metadata_group

def create_event_group(hf, event_id, condition, hr, event_timestamp, max_vital_history=30):
    """Create an event group with all signal data."""
    event_group = hf.create_group(f'event_{event_id}')

    # Generate all ECG leads
    ecg_lead_I = generate_ecg_lead(hr, condition, lead_type='I')
    ecg_lead_II = generate_ecg_lead(hr, condition, lead_type='II')
    ecg_lead_III = ecg_lead_II - ecg_lead_I  # Einthoven's law
    ecg_aVR = generate_ecg_lead(hr, condition, lead_type='aVR')
    ecg_aVL = generate_ecg_lead(hr, condition, lead_type='aVL')
    ecg_aVF = generate_ecg_lead(hr, condition, lead_type='aVF')
    ecg_vVX = generate_ecg_lead(hr, condition, lead_type='vVX')

    # ECG group with extras JSON
    ecg_group = event_group.create_group('ecg')
    ecg_group.create_dataset('ECG1', data=ecg_lead_I, compression='gzip')
    ecg_group.create_dataset('ECG2', data=ecg_lead_II, compression='gzip')
    ecg_group.create_dataset('ECG3', data=ecg_lead_III, compression='gzip')
    ecg_group.create_dataset('aVR', data=ecg_aVR, compression='gzip')
    ecg_group.create_dataset('aVL', data=ecg_aVL, compression='gzip')
    ecg_group.create_dataset('aVF', data=ecg_aVF, compression='gzip')
    ecg_group.create_dataset('vVX', data=ecg_vVX, compression='gzip')

    # Create extras JSON object for ECG with pacer information
    pacer_info = generate_pacer_info(condition)
    pacer_offset = generate_pacer_offset(condition)
    ecg_extras = {
        "pacer_info": int(pacer_info),
        "pacer_offset": int(pacer_offset)
    }
    ecg_group.create_dataset('extras', data=json.dumps(ecg_extras).encode('utf-8'))

    # PPG group with extras
    ppg_signal = generate_ppg_signal(hr, condition)
    ppg_group = event_group.create_group('ppg')
    ppg_group.create_dataset('PPG', data=ppg_signal, compression='gzip')

    # Create empty extras JSON object for PPG (user can customize later)
    ppg_extras = {}
    ppg_group.create_dataset('extras', data=json.dumps(ppg_extras).encode('utf-8'))

    # Respiratory group with extras
    resp_signal = generate_respiratory_waveform(hr, condition)
    resp_group = event_group.create_group('resp')
    resp_group.create_dataset('RESP', data=resp_signal, compression='gzip')

    # Create empty extras JSON object for respiratory (user can customize later)
    resp_extras = {}
    resp_group.create_dataset('extras', data=json.dumps(resp_extras).encode('utf-8'))

    # Vitals group (single values with units, timestamps, and extras)
    vitals_data = generate_vitals_single(hr, condition, event_timestamp)
    vitals_group = event_group.create_group('vitals')
    for vital_name, vital_info in vitals_data.items():
        vital_subgroup = vitals_group.create_group(vital_name)
        vital_subgroup.create_dataset('value', data=vital_info['value'])
        vital_subgroup.create_dataset('units', data=vital_info['units'].encode('utf-8'))
        vital_subgroup.create_dataset('timestamp', data=vital_info['timestamp'])

        # Create extras JSON object for each vital
        vital_extras = {}

        # Add thresholds to extras for all vitals except XL_Posture
        if vital_name != 'XL_Posture':
            vital_extras['upper_threshold'] = vital_info['upper_threshold']
            vital_extras['lower_threshold'] = vital_info['lower_threshold']
        else:
            # Add special attributes to extras for XL_Posture
            vital_extras['step_count'] = vital_info['step_count']
            vital_extras['time_since_posture_change'] = vital_info['time_since_posture_change']

        # Generate and add history data
        vital_history = generate_vital_history(
            vital_name,
            vital_info['value'],
            vital_info['timestamp'],
            condition,
            max_vital_history
        )
        vital_extras['history'] = vital_history

        # Store extras as JSON
        vital_subgroup.create_dataset('extras', data=json.dumps(vital_extras).encode('utf-8'))

    # Convert event timestamp to epoch
    event_epoch = time.mktime(event_timestamp.timetuple()) + event_timestamp.microsecond / 1e6

    # Single event timestamp for all signal types (epoch format)
    event_group.create_dataset('timestamp', data=event_epoch)

    # Generate and store event UUID
    event_uuid = str(uuid.uuid4())
    event_group.create_dataset('uuid', data=event_uuid)

    # Store condition as attribute
    event_group.attrs['condition'] = condition
    event_group.attrs['heart_rate'] = hr
    event_group.attrs['event_timestamp'] = event_epoch
    event_group.attrs['uuid'] = event_uuid

    return event_group

def main(num_events, patient_id=None, condition_proportions=None):
    """
    Main function to generate the event-based dataset.

    Args:
        num_events (int): Number of events to generate
        patient_id (str): Patient ID, generated if None
        condition_proportions (list): List of 5 proportions for [Normal, Tachycardia, Bradycardia, AFib, VTach]
    """
    if patient_id is None:
        patient_id = generate_patient_id()

    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    # Generate filename with current year-month
    current_date = datetime.now()
    filename = f"{patient_id}_{current_date.strftime('%Y-%m')}.h5"
    filepath = os.path.join(data_dir, filename)

    print(f"Generating {num_events} alarm events for patient {patient_id}")
    print(f"Output file: {filepath}")

    # Generate timestamps for all events
    event_timestamps = generate_event_timestamps(num_events)

    with h5py.File(filepath, 'w') as hf:
        # Create metadata group
        metadata_group = create_metadata_group(hf, patient_id, event_timestamps)
        print(f"  - Created metadata group")

        # Get max vital history from metadata
        max_vital_history = metadata_group['max_vital_history'][()]

        # Create event groups
        for i in range(num_events):
            condition, hr = generate_condition_and_hr(condition_proportions)
            event_timestamp = event_timestamps[i]

            print(f"  - Creating event_{1001+i}: {condition} (HR: {hr} bpm) at {event_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

            event_group = create_event_group(hf, 1001+i, condition, hr, event_timestamp, max_vital_history)

    print(f"\nDataset generation complete.")
    print(f"File '{filepath}' created successfully.")
    print("\nFile structure:")
    print(f"├── metadata/ (sampling rates, timing, units)")
    print(f"├── event_1001/ through event_{1000+num_events}/")
    print(f"    ├── ecg/ (7 leads at 200Hz + extras JSON with pacer info)")
    print(f"    ├── ppg/ (signal at 75Hz + extras JSON for customization)")
    print(f"    ├── resp/ (signal at 33.33Hz + extras JSON for customization)")
    print(f"    └── vitals/ (single values: HR, Pulse, SpO2, BP, Temp, Posture + extras JSON with thresholds)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a simulated patient dataset with alarm events in HDF5 format.",
        epilog="""
Examples:
  python rmsai_sim_hdf5_data.py 10                                    # Generate 10 events with default proportions (10%% normal, 90%% abnormal)
  python rmsai_sim_hdf5_data.py 20 --normal 50                        # Generate 20 events with 50%% normal events
  python rmsai_sim_hdf5_data.py 15 --normal 30 --tachy 20 --brady 20 --afib 20 --vtach 10  # Custom proportions
  python rmsai_sim_hdf5_data.py 25 --proportions 0.2 0.2 0.2 0.2 0.2 # Equal distribution across all conditions
  python rmsai_sim_hdf5_data.py 30 --all-normal                       # Generate only normal events
  python rmsai_sim_hdf5_data.py 40 --all-abnormal                     # Generate only abnormal events (equal mix)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'num_events',
        type=int,
        nargs='?',
        default=5,
        help="The number of alarm events to generate. Defaults to 5."
    )
    parser.add_argument(
        '--patient-id',
        type=str,
        help="Patient ID (e.g., PT1234). If not provided, will be randomly generated."
    )

    # Proportion configuration options
    proportion_group = parser.add_argument_group('condition proportions',
                                                'Configure the proportion of different cardiac conditions')

    proportion_group.add_argument(
        '--proportions',
        type=float,
        nargs=5,
        metavar=('NORMAL', 'TACHY', 'BRADY', 'AFIB', 'VTACH'),
        help="Specify exact proportions for all 5 conditions: Normal, Tachycardia, Bradycardia, AFib, VTach"
    )

    proportion_group.add_argument(
        '--normal',
        type=float,
        metavar='PCT',
        help="Percentage of normal events (0-100). Other conditions split equally."
    )

    proportion_group.add_argument(
        '--tachy',
        type=float,
        metavar='PCT',
        help="Percentage of tachycardia events (0-100)"
    )

    proportion_group.add_argument(
        '--brady',
        type=float,
        metavar='PCT',
        help="Percentage of bradycardia events (0-100)"
    )

    proportion_group.add_argument(
        '--afib',
        type=float,
        metavar='PCT',
        help="Percentage of atrial fibrillation events (0-100)"
    )

    proportion_group.add_argument(
        '--vtach',
        type=float,
        metavar='PCT',
        help="Percentage of ventricular tachycardia events (0-100)"
    )

    # Convenience options
    convenience_group = parser.add_argument_group('convenience options',
                                                 'Quick preset configurations')

    convenience_group.add_argument(
        '--all-normal',
        action='store_true',
        help="Generate only normal cardiac events"
    )

    convenience_group.add_argument(
        '--all-abnormal',
        action='store_true',
        help="Generate only abnormal events (equal mix of 4 abnormal conditions)"
    )

    convenience_group.add_argument(
        '--high-anomaly',
        action='store_true',
        help="Generate high anomaly dataset (5%% normal, 95%% abnormal)"
    )

    convenience_group.add_argument(
        '--balanced',
        action='store_true',
        help="Generate balanced dataset (equal 20%% for each condition)"
    )

    args = parser.parse_args()

    # Process proportion arguments
    condition_proportions = None

    if args.proportions:
        condition_proportions = args.proportions
    elif any([args.normal, args.tachy, args.brady, args.afib, args.vtach]):
        # Individual percentages specified
        proportions = [
            args.normal or 0,
            args.tachy or 0,
            args.brady or 0,
            args.afib or 0,
            args.vtach or 0
        ]

        # If only some conditions specified, distribute remainder equally
        specified_total = sum(p for p in proportions if p > 0)
        if specified_total > 100:
            parser.error("Total percentages cannot exceed 100%%")
        elif specified_total < 100:
            unspecified_count = sum(1 for p in proportions if p == 0)
            if unspecified_count > 0:
                remainder = (100 - specified_total) / unspecified_count
                proportions = [p if p > 0 else remainder for p in proportions]

        condition_proportions = [p / 100.0 for p in proportions]
    elif args.all_normal:
        condition_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    elif args.all_abnormal:
        condition_proportions = [0.0, 0.25, 0.25, 0.25, 0.25]
    elif args.high_anomaly:
        condition_proportions = [0.05, 0.2375, 0.2375, 0.2375, 0.2375]
    elif args.balanced:
        condition_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]

    # Display configuration
    if condition_proportions:
        condition_names = ['Normal', 'Tachycardia', 'Bradycardia', 'Atrial Fibrillation', 'Ventricular Tachycardia']
        print("Condition proportions:")
        for name, prop in zip(condition_names, condition_proportions):
            print(f"  {name}: {prop*100:.1f}%")
        print()

    main(args.num_events, args.patient_id, condition_proportions)