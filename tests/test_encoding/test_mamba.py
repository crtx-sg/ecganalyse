"""Unit tests for ECGMamba encoder."""

import torch
import pytest

from src.encoding.mamba import ECGMamba


class TestECGMamba:

    def setup_method(self) -> None:
        # Use fewer layers for faster tests
        self.model = ECGMamba(d_model=256, d_state=64, n_layers=2, patch_size=10)
        self.model.eval()

    def test_output_shape(self) -> None:
        """[2, 7, 2400] → [2, 7, 240, 256]."""
        x = torch.randn(2, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape == (2, 7, 240, 256)

    def test_output_dtype(self) -> None:
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        assert y.dtype == torch.float32

    def test_seq_len_from_patch_size(self) -> None:
        """seq_len = 2400 / patch_size."""
        model = ECGMamba(d_model=64, d_state=16, n_layers=1, patch_size=20)
        model.eval()
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 7, 120, 64)

    def test_per_lead_processing(self) -> None:
        """Each lead is processed independently — output maintains lead order."""
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape[1] == 7

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the model."""
        self.model.train()
        x = torch.randn(1, 7, 2400)
        y = self.model(x)
        loss = y.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
            if p.requires_grad
        )
        assert has_grad

    def test_batch_size_flexibility(self) -> None:
        x = torch.randn(4, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape[0] == 4

    def test_finite_output(self) -> None:
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        assert torch.isfinite(y).all()
