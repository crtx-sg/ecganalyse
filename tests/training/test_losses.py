"""Tests for training loss functions."""

import torch
import pytest

from src.training.losses import DenoiserLoss, HeatmapLoss


class TestDenoiserLoss:
    def test_zero_loss_for_identical_inputs(self):
        criterion = DenoiserLoss()
        x = torch.randn(2, 7, 2400)
        loss = criterion(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_for_different_inputs(self):
        criterion = DenoiserLoss()
        pred = torch.randn(2, 7, 2400)
        target = torch.randn(2, 7, 2400)
        loss = criterion(pred, target)
        assert loss.item() > 0.0

    def test_loss_is_scalar(self):
        criterion = DenoiserLoss()
        pred = torch.randn(2, 7, 2400)
        target = torch.randn(2, 7, 2400)
        loss = criterion(pred, target)
        assert loss.shape == ()

    def test_gradients_flow(self):
        criterion = DenoiserLoss()
        pred = torch.randn(2, 7, 2400, requires_grad=True)
        target = torch.randn(2, 7, 2400)
        loss = criterion(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_alpha_beta_weighting(self):
        pred = torch.randn(2, 7, 2400)
        target = torch.randn(2, 7, 2400)

        loss_default = DenoiserLoss(alpha=1.0, beta=0.1)(pred, target)
        loss_no_spectral = DenoiserLoss(alpha=1.0, beta=0.0)(pred, target)
        # With spectral component, loss should differ
        assert loss_default.item() != pytest.approx(loss_no_spectral.item(), abs=1e-6)


class TestHeatmapLoss:
    def test_loss_is_positive(self):
        criterion = HeatmapLoss()
        logits = torch.randn(2, 7, 9, 2400)
        targets = torch.zeros(2, 7, 9, 2400)
        loss = criterion(logits, targets)
        assert loss.item() > 0.0

    def test_loss_is_scalar(self):
        criterion = HeatmapLoss()
        logits = torch.randn(2, 7, 9, 2400)
        targets = torch.rand(2, 7, 9, 2400)
        loss = criterion(logits, targets)
        assert loss.shape == ()

    def test_gradients_flow(self):
        criterion = HeatmapLoss()
        logits = torch.randn(2, 7, 9, 2400, requires_grad=True)
        targets = torch.rand(2, 7, 9, 2400)
        loss = criterion(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_perfect_prediction_low_loss(self):
        """When prediction is very confident and correct, loss should be lower."""
        criterion = HeatmapLoss()
        # All-zero targets with strongly negative logits (confident zeros)
        targets = torch.zeros(1, 7, 9, 2400)
        good_logits = torch.full((1, 7, 9, 2400), -10.0)
        bad_logits = torch.full((1, 7, 9, 2400), 10.0)

        good_loss = criterion(good_logits, targets)
        bad_loss = criterion(bad_logits, targets)
        assert good_loss.item() < bad_loss.item()

    def test_pos_weight_affects_loss(self):
        logits = torch.randn(1, 7, 9, 2400)
        targets = torch.rand(1, 7, 9, 2400)

        loss_high_pw = HeatmapLoss(pos_weight=100.0)(logits, targets)
        loss_low_pw = HeatmapLoss(pos_weight=1.0)(logits, targets)
        # Higher pos_weight should increase loss for positive targets
        assert loss_high_pw.item() != pytest.approx(loss_low_pw.item(), abs=0.01)
