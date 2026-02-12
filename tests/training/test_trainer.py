"""Smoke tests for ECG trainer — runs 1-2 batches per stage."""

import pytest
import torch

from src.training.trainer import ECGTrainer


@pytest.fixture
def trainer(tmp_path):
    return ECGTrainer(
        output_dir=str(tmp_path / "weights"),
        device="cpu",
        use_amp=False,
    )


class TestTrainerStageA:
    def test_smoke_stage_a(self, trainer):
        """Stage A should complete 2 epochs with tiny dataset."""
        best_path = trainer.train_stage_a(
            epochs=2, batch_size=2, epoch_size=4, val_size=4, patience=100,
        )
        assert best_path.exists()

    def test_stage_a_produces_checkpoint(self, trainer):
        trainer.train_stage_a(
            epochs=1, batch_size=2, epoch_size=4, val_size=4, patience=100,
        )
        assert (trainer.output_dir / "stage_a_latest.pt").exists()


class TestTrainerStageB:
    def test_smoke_stage_b(self, trainer):
        """Stage B should complete 1 epoch with tiny dataset."""
        best_path = trainer.train_stage_b(
            epochs=1, batch_size=2, epoch_size=4, val_size=4, patience=100,
        )
        assert best_path.exists()


class TestTrainerStageC:
    def test_smoke_stage_c(self, trainer):
        """Stage C should complete 1 epoch with tiny dataset."""
        best_path = trainer.train_stage_c(
            epochs=1, batch_size=2, epoch_size=4, val_size=4, patience=100,
        )
        assert best_path.exists()


class TestTrainerResume:
    def test_resume_stage_a(self, trainer):
        """Should be able to resume training from a checkpoint."""
        trainer.train_stage_a(
            epochs=1, batch_size=2, epoch_size=4, val_size=4, patience=100,
        )
        latest = trainer.output_dir / "stage_a_latest.pt"
        assert latest.exists()

        # Resume from checkpoint
        trainer.train_stage_a(
            epochs=2, batch_size=2, epoch_size=4, val_size=4,
            patience=100, resume_path=str(latest),
        )
