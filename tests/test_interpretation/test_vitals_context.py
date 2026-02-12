"""Unit tests for VitalsContextIntegrator."""

import pytest

from src.ecg_system.schemas import (
    Finding,
    GlobalMeasurements,
    VitalMeasurement,
    VitalsData,
)
from src.interpretation.vitals_context import VitalsContextIntegrator


def _measurements(hr: float = 72.0) -> GlobalMeasurements:
    return GlobalMeasurements(
        heart_rate_bpm=hr,
        rr_mean_ms=60000 / hr if hr > 0 else 0,
        rr_std_ms=30.0,
        rr_min_ms=700,
        rr_max_ms=900,
        pr_interval_ms=160.0,
        pr_interval_range_ms=(155.0, 165.0),
        qrs_duration_ms=85.0,
        qrs_duration_range_ms=(82.0, 88.0),
        qt_interval_ms=380.0,
        qtc_bazett_ms=416.0,
        qtc_fridericia_ms=408.0,
    )


def _vital(name: str, value: float, units: str, upper: float, lower: float) -> VitalMeasurement:
    return VitalMeasurement(
        name=name, value=value, units=units, timestamp=1704537600.0,
        extras={"upper_threshold": upper, "lower_threshold": lower},
    )


def _normal_vitals() -> VitalsData:
    return VitalsData(
        hr=_vital("hr", 73, "bpm", 120, 50),
        pulse=_vital("pulse", 74, "bpm", 120, 50),
        spo2=_vital("spo2", 96, "%", 100, 90),
        systolic=_vital("systolic", 125, "mmHg", 160, 90),
        diastolic=_vital("diastolic", 78, "mmHg", 100, 60),
        resp_rate=_vital("resp_rate", 16, "brpm", 30, 8),
        temp=_vital("temp", 37.1, "C", 38.5, 35.5),
    )


class TestHRValidation:
    """Tests for ECG vs Monitor HR validation."""

    def test_consistent_hr(self) -> None:
        integrator = VitalsContextIntegrator()
        findings = integrator.integrate(_measurements(hr=72), _normal_vitals())

        hr_findings = [f for f in findings if "HR" in f.finding and "consistent" in f.finding.lower()]
        assert len(hr_findings) == 1
        assert hr_findings[0].severity == "normal"

    def test_discrepant_hr(self) -> None:
        integrator = VitalsContextIntegrator()
        vitals = _normal_vitals()
        vitals.hr = _vital("hr", 95, "bpm", 120, 50)
        findings = integrator.integrate(_measurements(hr=72), vitals)

        hr_findings = [f for f in findings if "discrepancy" in f.finding.lower() or "HR" in f.finding]
        discrepant = [f for f in hr_findings if f.severity in ("moderate", "critical")]
        assert len(discrepant) >= 1

    def test_critical_hr_discrepancy(self) -> None:
        integrator = VitalsContextIntegrator()
        vitals = _normal_vitals()
        vitals.hr = _vital("hr", 120, "bpm", 150, 50)
        findings = integrator.integrate(_measurements(hr=72), vitals)

        critical = [f for f in findings if f.severity == "critical"]
        assert len(critical) >= 1

    def test_no_hr_vital(self) -> None:
        integrator = VitalsContextIntegrator()
        vitals = _normal_vitals()
        vitals.hr = None
        findings = integrator.integrate(_measurements(hr=72), vitals)
        # Should not crash, just no HR validation finding
        assert all(isinstance(f, Finding) for f in findings)


class TestThresholdChecks:
    """Tests for vital sign threshold violations."""

    def test_all_normal(self) -> None:
        integrator = VitalsContextIntegrator()
        findings = integrator.integrate(_measurements(), _normal_vitals())

        normal = [f for f in findings if "within normal" in f.finding.lower()]
        assert len(normal) == 1

    def test_spo2_below(self) -> None:
        integrator = VitalsContextIntegrator()
        vitals = _normal_vitals()
        vitals.spo2 = _vital("spo2", 85, "%", 100, 90)
        findings = integrator.integrate(_measurements(), vitals)

        violations = [f for f in findings if "below threshold" in f.finding]
        assert len(violations) >= 1
        assert any("spo2" in f.finding for f in violations)

    def test_hr_above(self) -> None:
        integrator = VitalsContextIntegrator()
        vitals = _normal_vitals()
        vitals.hr = _vital("hr", 130, "bpm", 120, 50)
        findings = integrator.integrate(_measurements(hr=130), vitals)

        violations = [f for f in findings if "above threshold" in f.finding]
        assert len(violations) >= 1

    def test_multiple_violations(self) -> None:
        integrator = VitalsContextIntegrator()
        vitals = _normal_vitals()
        vitals.spo2 = _vital("spo2", 85, "%", 100, 90)
        vitals.systolic = _vital("systolic", 180, "mmHg", 160, 90)
        findings = integrator.integrate(_measurements(), vitals)

        violations = [f for f in findings if "threshold" in f.finding]
        assert len(violations) >= 2
        # No "all normal" when violations exist
        normal = [f for f in findings if "within normal" in f.finding.lower()]
        assert len(normal) == 0


class TestCrossModalFindings:
    """Tests for cross-modal clinical correlations."""

    def test_tachy_hypertension(self) -> None:
        integrator = VitalsContextIntegrator()
        vitals = _normal_vitals()
        vitals.systolic = _vital("systolic", 150, "mmHg", 160, 90)
        findings = integrator.integrate(_measurements(hr=110), vitals)

        stress = [f for f in findings if "stress" in f.finding.lower()]
        assert len(stress) >= 1

    def test_brady_hypotension(self) -> None:
        integrator = VitalsContextIntegrator()
        vitals = _normal_vitals()
        vitals.systolic = _vital("systolic", 85, "mmHg", 160, 90)
        findings = integrator.integrate(_measurements(hr=45), vitals)

        hemo = [f for f in findings if "hemodynamic" in f.finding.lower()]
        assert len(hemo) >= 1

    def test_tachy_hypoxemia(self) -> None:
        integrator = VitalsContextIntegrator()
        vitals = _normal_vitals()
        vitals.spo2 = _vital("spo2", 85, "%", 100, 90)
        findings = integrator.integrate(_measurements(hr=110), vitals)

        hypox = [f for f in findings if "hypoxemia" in f.finding.lower()]
        assert len(hypox) >= 1

    def test_no_vitals(self) -> None:
        integrator = VitalsContextIntegrator()
        findings = integrator.integrate(_measurements(), None)

        assert len(findings) == 1
        assert "No vitals" in findings[0].finding
