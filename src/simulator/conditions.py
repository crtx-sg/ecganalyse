"""Cardiac condition definitions and morphology configurations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Condition(Enum):
    """Cardiac conditions mapped to MIT-BIH-style annotation codes."""

    # Normal rhythms
    NORMAL_SINUS = "N"
    SINUS_BRADYCARDIA = "SB"
    SINUS_TACHYCARDIA = "ST"

    # Supraventricular
    ATRIAL_FIBRILLATION = "AFIB"
    ATRIAL_FLUTTER = "AFL"
    PAC = "A"
    SVT = "SVTA"

    # Ventricular
    PVC = "V"
    VENTRICULAR_TACHYCARDIA = "VT"
    VENTRICULAR_FIBRILLATION = "VF"

    # Bundle branch blocks
    LBBB = "L"
    RBBB = "R"

    # AV blocks
    AV_BLOCK_1 = "1AVB"
    AV_BLOCK_2_TYPE1 = "2AVB1"
    AV_BLOCK_2_TYPE2 = "2AVB2"

    # Other
    ST_ELEVATION = "STE"


@dataclass(frozen=True)
class ConditionConfig:
    """Morphology configuration for a cardiac condition.

    Attributes:
        hr_range: (min, max) heart rate in BPM.
        qrs_duration_range: (min, max) QRS duration in seconds.
        p_wave_presence: probability that a P wave is present before each beat.
        t_wave_inversion_prob: probability that the T wave is inverted.
        rr_irregularity: standard deviation of RR jitter as fraction of beat interval.
        wide_qrs: whether the QRS complex should be widened.
        st_elevation: whether ST segment elevation is present.
    """

    hr_range: tuple[float, float]
    qrs_duration_range: tuple[float, float]
    p_wave_presence: float
    t_wave_inversion_prob: float
    rr_irregularity: float
    wide_qrs: bool
    st_elevation: bool


CONDITION_REGISTRY: dict[Condition, ConditionConfig] = {
    # --- Normal rhythms ---
    Condition.NORMAL_SINUS: ConditionConfig(
        hr_range=(60.0, 100.0),
        qrs_duration_range=(0.08, 0.12),
        p_wave_presence=0.95,
        t_wave_inversion_prob=0.05,
        rr_irregularity=0.03,
        wide_qrs=False,
        st_elevation=False,
    ),
    Condition.SINUS_BRADYCARDIA: ConditionConfig(
        hr_range=(40.0, 59.0),
        qrs_duration_range=(0.08, 0.12),
        p_wave_presence=0.95,
        t_wave_inversion_prob=0.05,
        rr_irregularity=0.03,
        wide_qrs=False,
        st_elevation=False,
    ),
    Condition.SINUS_TACHYCARDIA: ConditionConfig(
        hr_range=(101.0, 150.0),
        qrs_duration_range=(0.08, 0.12),
        p_wave_presence=0.90,
        t_wave_inversion_prob=0.10,
        rr_irregularity=0.02,
        wide_qrs=False,
        st_elevation=False,
    ),
    # --- Supraventricular ---
    Condition.ATRIAL_FIBRILLATION: ConditionConfig(
        hr_range=(80.0, 180.0),
        qrs_duration_range=(0.08, 0.12),
        p_wave_presence=0.10,
        t_wave_inversion_prob=0.20,
        rr_irregularity=0.25,
        wide_qrs=False,
        st_elevation=False,
    ),
    Condition.ATRIAL_FLUTTER: ConditionConfig(
        hr_range=(75.0, 150.0),
        qrs_duration_range=(0.08, 0.12),
        p_wave_presence=0.05,
        t_wave_inversion_prob=0.15,
        rr_irregularity=0.05,
        wide_qrs=False,
        st_elevation=False,
    ),
    Condition.PAC: ConditionConfig(
        hr_range=(60.0, 100.0),
        qrs_duration_range=(0.08, 0.12),
        p_wave_presence=0.85,
        t_wave_inversion_prob=0.10,
        rr_irregularity=0.15,
        wide_qrs=False,
        st_elevation=False,
    ),
    Condition.SVT: ConditionConfig(
        hr_range=(150.0, 250.0),
        qrs_duration_range=(0.08, 0.12),
        p_wave_presence=0.30,
        t_wave_inversion_prob=0.15,
        rr_irregularity=0.02,
        wide_qrs=False,
        st_elevation=False,
    ),
    # --- Ventricular ---
    Condition.PVC: ConditionConfig(
        hr_range=(60.0, 100.0),
        qrs_duration_range=(0.12, 0.20),
        p_wave_presence=0.20,
        t_wave_inversion_prob=0.60,
        rr_irregularity=0.20,
        wide_qrs=True,
        st_elevation=False,
    ),
    Condition.VENTRICULAR_TACHYCARDIA: ConditionConfig(
        hr_range=(110.0, 200.0),
        qrs_duration_range=(0.12, 0.20),
        p_wave_presence=0.20,
        t_wave_inversion_prob=0.60,
        rr_irregularity=0.03,
        wide_qrs=True,
        st_elevation=False,
    ),
    Condition.VENTRICULAR_FIBRILLATION: ConditionConfig(
        hr_range=(150.0, 500.0),
        qrs_duration_range=(0.06, 0.30),
        p_wave_presence=0.0,
        t_wave_inversion_prob=0.50,
        rr_irregularity=0.40,
        wide_qrs=True,
        st_elevation=False,
    ),
    # --- Bundle branch blocks ---
    Condition.LBBB: ConditionConfig(
        hr_range=(60.0, 100.0),
        qrs_duration_range=(0.12, 0.18),
        p_wave_presence=0.90,
        t_wave_inversion_prob=0.70,
        rr_irregularity=0.03,
        wide_qrs=True,
        st_elevation=False,
    ),
    Condition.RBBB: ConditionConfig(
        hr_range=(60.0, 100.0),
        qrs_duration_range=(0.12, 0.16),
        p_wave_presence=0.90,
        t_wave_inversion_prob=0.40,
        rr_irregularity=0.03,
        wide_qrs=True,
        st_elevation=False,
    ),
    # --- AV blocks ---
    Condition.AV_BLOCK_1: ConditionConfig(
        hr_range=(55.0, 100.0),
        qrs_duration_range=(0.08, 0.12),
        p_wave_presence=0.95,
        t_wave_inversion_prob=0.05,
        rr_irregularity=0.03,
        wide_qrs=False,
        st_elevation=False,
    ),
    Condition.AV_BLOCK_2_TYPE1: ConditionConfig(
        hr_range=(45.0, 80.0),
        qrs_duration_range=(0.08, 0.12),
        p_wave_presence=0.95,
        t_wave_inversion_prob=0.05,
        rr_irregularity=0.15,
        wide_qrs=False,
        st_elevation=False,
    ),
    Condition.AV_BLOCK_2_TYPE2: ConditionConfig(
        hr_range=(40.0, 70.0),
        qrs_duration_range=(0.08, 0.14),
        p_wave_presence=0.95,
        t_wave_inversion_prob=0.10,
        rr_irregularity=0.20,
        wide_qrs=False,
        st_elevation=False,
    ),
    # --- Other ---
    Condition.ST_ELEVATION: ConditionConfig(
        hr_range=(60.0, 120.0),
        qrs_duration_range=(0.08, 0.12),
        p_wave_presence=0.90,
        t_wave_inversion_prob=0.30,
        rr_irregularity=0.03,
        wide_qrs=False,
        st_elevation=True,
    ),
}
