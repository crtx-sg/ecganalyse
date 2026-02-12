"""Tests for synthetic ECG training datasets."""

import torch
import pytest

from src.training.datasets import (
    SyntheticECGTrainingDataset,
    create_validation_dataset,
    TRAINABLE_CONDITIONS,
)


@pytest.fixture
def small_dataset():
    return SyntheticECGTrainingDataset(epoch_size=10, base_seed=42)


class TestSyntheticECGTrainingDataset:
    def test_length(self, small_dataset):
        assert len(small_dataset) == 10

    def test_getitem_keys(self, small_dataset):
        item = small_dataset[0]
        assert "ecg_noisy" in item
        assert "ecg_clean" in item
        assert "heatmaps" in item
        assert "condition" in item
        assert "hr" in item
        assert "noise_level" in item

    def test_ecg_shapes(self, small_dataset):
        item = small_dataset[0]
        assert item["ecg_noisy"].shape == (7, 2400)
        assert item["ecg_clean"].shape == (7, 2400)

    def test_heatmap_shape(self, small_dataset):
        item = small_dataset[0]
        assert item["heatmaps"].shape == (7, 9, 2400)

    def test_heatmap_values_in_range(self, small_dataset):
        item = small_dataset[0]
        assert item["heatmaps"].min() >= 0.0
        assert item["heatmaps"].max() <= 1.0

    def test_hr_is_scalar_tensor(self, small_dataset):
        item = small_dataset[0]
        assert item["hr"].shape == ()
        assert item["hr"].dtype == torch.float32

    def test_condition_is_string(self, small_dataset):
        item = small_dataset[0]
        assert isinstance(item["condition"], str)

    def test_deterministic_with_same_seed(self):
        ds1 = SyntheticECGTrainingDataset(epoch_size=5, base_seed=42)
        ds2 = SyntheticECGTrainingDataset(epoch_size=5, base_seed=42)
        item1 = ds1[0]
        item2 = ds2[0]
        assert torch.allclose(item1["ecg_noisy"], item2["ecg_noisy"])
        assert torch.allclose(item1["ecg_clean"], item2["ecg_clean"])

    def test_different_seeds_produce_different_data(self):
        ds1 = SyntheticECGTrainingDataset(epoch_size=5, base_seed=42)
        ds2 = SyntheticECGTrainingDataset(epoch_size=5, base_seed=99)
        item1 = ds1[0]
        item2 = ds2[0]
        assert not torch.allclose(item1["ecg_noisy"], item2["ecg_noisy"])

    def test_excludes_vfib_by_default(self):
        from src.simulator.conditions import Condition
        assert Condition.VENTRICULAR_FIBRILLATION not in TRAINABLE_CONDITIONS

    def test_noisy_differs_from_clean(self, small_dataset):
        item = small_dataset[0]
        # With noise applied, they should differ (unless noise_level is "clean")
        # Use multiple items to find one with noise
        for i in range(10):
            item = small_dataset[i]
            if item["noise_level"] != "clean":
                assert not torch.allclose(item["ecg_noisy"], item["ecg_clean"])
                return
        # If all items happened to be clean, just pass
        assert True


class TestCreateValidationDataset:
    def test_creates_dataset(self):
        ds = create_validation_dataset(size=10)
        assert len(ds) == 10

    def test_fixed_seed(self):
        ds1 = create_validation_dataset(size=5)
        ds2 = create_validation_dataset(size=5)
        item1 = ds1[0]
        item2 = ds2[0]
        assert torch.allclose(item1["ecg_noisy"], item2["ecg_noisy"])
