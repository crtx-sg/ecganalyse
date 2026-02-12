"""Gate tests for Phase 4: Neuro-Symbolic Interpretation output contracts."""

import json

import numpy as np
import torch
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.encoding.foundation import FoundationModelAdapter
from src.prediction.heatmap import HeatmapDecoder
from src.prediction.fiducial import FiducialExtractor
from src.preprocessing.quality import SignalQualityAssessor
from src.interpretation.symbolic import SymbolicCalculationEngine
from src.interpretation.rules import RuleBasedReasoningEngine
from src.interpretation.vitals_context import VitalsContextIntegrator
from src.interpretation.assembly import JSONAssembler

pytestmark = pytest.mark.gate

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version", "generated_at", "processing_time_ms",
    "event_context", "metadata", "vitals_context", "quality",
    "global_measurements", "rhythm", "beats", "findings",
    "calculation_traces", "summary",
}


class TestPhase4OutputContract:
    """Validate JSON Feature Assembly output contract."""

    def _run_pipeline(self, sample_hdf5_path: str) -> dict:
        """Run full Phase 0-4 pipeline and return JSON assembly."""
        loader = HDF5AlarmEventLoader()
        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, "event_1001")

        # Phase 1: Quality
        assessor = SignalQualityAssessor()
        quality = assessor.assess(event.ecg)

        # Phase 2-3: Encode + Decode + Extract
        encoder = FoundationModelAdapter(output_dim=256)
        encoder.eval()
        decoder = HeatmapDecoder(d_model=256)
        decoder.eval()
        extractor = FiducialExtractor()

        ecg_tensor = torch.from_numpy(event.ecg.as_array).unsqueeze(0).float()
        with torch.no_grad():
            features = encoder(ecg_tensor)
            heatmaps = decoder(features)
        beats = extractor.extract(heatmaps.squeeze(0).numpy(), event.ecg.as_array)

        # Phase 4: Interpretation
        sym = SymbolicCalculationEngine(fs=200)
        measurements = sym.compute_global_measurements(beats)
        rhythm_metrics = sym.compute_rhythm_metrics(beats)

        rules = RuleBasedReasoningEngine()
        rhythm = rules.classify_rhythm(measurements, rhythm_metrics, beats)
        findings = rules.generate_findings(measurements, rhythm, beats)

        vitals_integrator = VitalsContextIntegrator()
        vital_findings = vitals_integrator.integrate(measurements, event.vitals)
        findings.extend(vital_findings)

        assembler = JSONAssembler()
        return assembler.assemble(
            event, quality, measurements, rhythm, beats,
            findings, sym.traces,
        )

    def test_schema_version(self, sample_hdf5_path: str) -> None:
        result = self._run_pipeline(sample_hdf5_path)
        assert result["schema_version"] == "1.1"

    def test_required_keys_present(self, sample_hdf5_path: str) -> None:
        result = self._run_pipeline(sample_hdf5_path)
        assert REQUIRED_TOP_LEVEL_KEYS.issubset(result.keys())

    def test_json_serializable(self, sample_hdf5_path: str) -> None:
        result = self._run_pipeline(sample_hdf5_path)
        json_str = json.dumps(result, indent=2)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.1"

    def test_event_context_preserved(self, sample_hdf5_path: str) -> None:
        result = self._run_pipeline(sample_hdf5_path)
        ctx = result["event_context"]
        assert ctx["event_id"] == "event_1001"
        assert "event_uuid" in ctx
        assert "event_timestamp" in ctx

    def test_findings_not_empty(self, sample_hdf5_path: str) -> None:
        result = self._run_pipeline(sample_hdf5_path)
        assert len(result["findings"]) >= 1
        categories = {f["category"] for f in result["findings"]}
        assert "rhythm" in categories

    def test_calculation_traces_present(self, sample_hdf5_path: str) -> None:
        result = self._run_pipeline(sample_hdf5_path)
        assert len(result["calculation_traces"]) > 0

    def test_rhythm_has_classification(self, sample_hdf5_path: str) -> None:
        result = self._run_pipeline(sample_hdf5_path)
        rhythm = result["rhythm"]
        assert "classification" in rhythm
        assert "classification_confidence" in rhythm
        assert 0.0 <= rhythm["classification_confidence"] <= 1.0

    def test_summary_present(self, sample_hdf5_path: str) -> None:
        result = self._run_pipeline(sample_hdf5_path)
        summary = result["summary"]
        assert "primary_interpretation" in summary
        assert isinstance(summary["categories_present"], list)
        assert isinstance(summary["abnormality_count"], int)
        assert isinstance(summary["critical_findings"], list)

    def test_global_measurements_valid(self, sample_hdf5_path: str) -> None:
        result = self._run_pipeline(sample_hdf5_path)
        gm = result["global_measurements"]
        assert "heart_rate_bpm" in gm
        assert "rr_intervals_ms" in gm
        assert "qtc_bazett_ms" in gm

    def test_vitals_context_present(self, sample_hdf5_path: str) -> None:
        result = self._run_pipeline(sample_hdf5_path)
        # Sample HDF5 has vitals
        assert result["vitals_context"] is not None
