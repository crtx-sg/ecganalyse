"""Unit tests for HeatmapDecoder."""

import torch
import pytest

from src.prediction.heatmap import HeatmapDecoder, NUM_FIDUCIAL_TYPES


class TestHeatmapDecoder:

    def setup_method(self) -> None:
        self.model = HeatmapDecoder(d_model=256, base_channels=64)
        self.model.eval()

    def test_output_shape(self) -> None:
        """[B, 7, 9, 2400] output for [B, 7, 240, 256] input."""
        x = torch.randn(1, 7, 240, 256)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape == (1, 7, 9, 2400)

    def test_output_range(self) -> None:
        """Heatmap values should be in [0, 1] (sigmoid output)."""
        x = torch.randn(1, 7, 240, 256)
        with torch.no_grad():
            y = self.model(x)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_num_fiducial_types(self) -> None:
        assert NUM_FIDUCIAL_TYPES == 9

    def test_batch_dimension(self) -> None:
        x = torch.randn(4, 7, 240, 256)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape == (4, 7, 9, 2400)

    def test_gradient_flow(self) -> None:
        self.model.train()
        x = torch.randn(1, 7, 240, 256)
        y = self.model(x)
        loss = y.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
        )
        assert has_grad

    def test_different_seq_len(self) -> None:
        """Should work with different input sequence lengths."""
        x = torch.randn(1, 7, 120, 256)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape == (1, 7, 9, 2400)  # still outputs 2400

    def test_finite_output(self) -> None:
        x = torch.randn(1, 7, 240, 256)
        with torch.no_grad():
            y = self.model(x)
        assert torch.isfinite(y).all()
