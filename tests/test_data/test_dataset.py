"""Unit tests for ECGAlarmDataset."""

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.data.dataset import ECGAlarmDataset


@pytest.fixture
def dataset_dir(sample_hdf5_path: str, tmp_path: Path) -> str:
    """Create a directory with a copy of the sample HDF5 for dataset tests."""
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir()
    shutil.copy2(sample_hdf5_path, ds_dir / "test_patient_2024-01.h5")
    return str(ds_dir)


class TestECGAlarmDataset:
    def test_dataset_length(self, dataset_dir: str) -> None:
        ds = ECGAlarmDataset(dataset_dir)
        assert len(ds) == 2  # 2 events in sample file

    def test_dataset_getitem_ecg_shape(self, dataset_dir: str) -> None:
        ds = ECGAlarmDataset(dataset_dir)
        sample = ds[0]
        assert "ecg" in sample
        assert isinstance(sample["ecg"], torch.Tensor)
        assert sample["ecg"].shape == (7, 2400)
        assert sample["ecg"].dtype == torch.float32

    def test_dataset_getitem_metadata(self, dataset_dir: str) -> None:
        ds = ECGAlarmDataset(dataset_dir)
        sample = ds[0]
        assert "event_id" in sample
        assert "patient_id" in sample
        assert sample["patient_id"] == "PT_TEST"
        assert sample["event_id"].startswith("event_")
        assert "metadata" in sample
        assert sample["metadata"]["sample_rate"] == 200

    def test_dataset_with_vitals(self, dataset_dir: str) -> None:
        ds = ECGAlarmDataset(dataset_dir, include_vitals=True)
        sample = ds[0]
        assert "vitals" in sample
        assert "hr" in sample["vitals"]

    def test_dataset_without_vitals(self, dataset_dir: str) -> None:
        ds = ECGAlarmDataset(dataset_dir, include_vitals=False)
        sample = ds[0]
        assert sample["vitals"] == {}

    def test_dataset_with_auxiliary(self, dataset_dir: str) -> None:
        ds = ECGAlarmDataset(dataset_dir, include_auxiliary=True)
        sample = ds[0]
        assert "ppg" in sample
        assert isinstance(sample["ppg"], torch.Tensor)
        assert sample["ppg"].shape == (900,)
        assert "resp" in sample
        assert isinstance(sample["resp"], torch.Tensor)
        assert sample["resp"].shape == (400,)

    def test_dataset_without_auxiliary(self, dataset_dir: str) -> None:
        ds = ECGAlarmDataset(dataset_dir, include_auxiliary=False)
        sample = ds[0]
        assert "ppg" not in sample
        assert "resp" not in sample

    def test_get_event_by_id(self, dataset_dir: str) -> None:
        ds = ECGAlarmDataset(dataset_dir)
        sample = ds.get_event_by_id("PT_TEST", "event_1001")
        assert sample is not None
        assert sample["event_id"] == "event_1001"

    def test_get_event_by_id_not_found(self, dataset_dir: str) -> None:
        ds = ECGAlarmDataset(dataset_dir)
        sample = ds.get_event_by_id("NONEXISTENT", "event_9999")
        assert sample is None

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        ds = ECGAlarmDataset(str(empty_dir))
        assert len(ds) == 0

    def test_transform_applied(self, dataset_dir: str) -> None:
        def add_flag(sample: dict) -> dict:
            sample["transformed"] = True
            return sample

        ds = ECGAlarmDataset(dataset_dir, transform=add_flag)
        sample = ds[0]
        assert sample["transformed"] is True
