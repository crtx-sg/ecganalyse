"""Integration tests for Phase 4: beats → interpretation → JSON assembly."""

import json

import numpy as np
import torch
import pytest

from src.data.hdf5_loader import HDF5AlarmEventLoader
from src.preprocessing.quality import SignalQualityAssessor
from src.encoding.foundation import FoundationModelAdapter
from src.prediction.heatmap import HeatmapDecoder
from src.prediction.fiducial import FiducialExtractor
from src.interpretation.symbolic import SymbolicCalculationEngine
from src.interpretation.rules import RuleBasedReasoningEngine
from src.interpretation.vitals_context import VitalsContextIntegrator
from src.interpretation.assembly import JSONAssembler

pytestmark = pytest.mark.integration


class TestPhase4Integration:
    """End-to-end: HDF5 → Phase 0-3 → Phase 4 interpretation."""

    def _full_pipeline(self, sample_hdf5_path: str, event_id: str = "event_1001") -> dict:
        loader = HDF5AlarmEventLoader()
        with loader.load_file(sample_hdf5_path) as f:
            event = loader.load_event(f, event_id)

        assessor = SignalQualityAssessor()
        quality = assessor.assess(event.ecg)

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

    def test_full_pipeline_produces_valid_json(self, sample_hdf5_path: str) -> None:
        """Full pipeline produces JSON-serializable output."""
        result = self._full_pipeline(sample_hdf5_path)
        json_str = json.dumps(result, indent=2)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.1"

    def test_pipeline_with_multiple_events(self, sample_hdf5_path: str) -> None:
        """Pipeline works on multiple events."""
        for eid in ["event_1001", "event_1002"]:
            result = self._full_pipeline(sample_hdf5_path, eid)
            assert result["event_context"]["event_id"] == eid

    def test_pipeline_no_vitals(self, sample_hdf5_no_vitals: str) -> None:
        """Pipeline handles missing vitals gracefully."""
        loader = HDF5AlarmEventLoader()
        with loader.load_file(sample_hdf5_no_vitals) as f:
            event = loader.load_event(f, "event_1001")

        assessor = SignalQualityAssessor()
        quality = assessor.assess(event.ecg)

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
        result = assembler.assemble(
            event, quality, measurements, rhythm, beats,
            findings, sym.traces,
        )
        assert result["vitals_context"] is None
        assert any("No vitals" in f["finding"] for f in result["findings"])

    def test_findings_categories(self, sample_hdf5_path: str) -> None:
        """Output contains both rhythm and vital findings."""
        result = self._full_pipeline(sample_hdf5_path)
        categories = {f["category"] for f in result["findings"]}
        assert "rhythm" in categories
        assert "vital" in categories

    def test_summary_consistency(self, sample_hdf5_path: str) -> None:
        """Summary fields are consistent with findings."""
        result = self._full_pipeline(sample_hdf5_path)
        summary = result["summary"]
        findings = result["findings"]
        categories = sorted(set(f["category"] for f in findings))
        assert summary["categories_present"] == categories
