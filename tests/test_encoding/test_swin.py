"""Unit tests for Swin1DTransformer encoder."""

import torch
import pytest

from src.encoding.swin import Swin1DTransformer


class TestSwin1DTransformer:

    def setup_method(self) -> None:
        self.model = Swin1DTransformer(
            embed_dim=128,
            depths=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            window_sizes=[20, 100, 400, 2400],
            patch_size=4,
        )
        self.model.eval()

    def test_output_shape(self) -> None:
        """[2, 7, 2400] → [2, 7, seq_len, output_dim]."""
        x = torch.randn(2, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        B, leads, S, D = y.shape
        assert B == 2
        assert leads == 7
        assert S > 0
        assert D == self.model.output_dim

    def test_output_dtype(self) -> None:
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        assert y.dtype == torch.float32

    def test_hierarchical_stages(self) -> None:
        """Model should have 4 stages matching the 4 hierarchical levels."""
        assert len(self.model.stages) == 4

    def test_window_sizes_configured(self) -> None:
        """Configured window sizes should be [20, 100, 400, 2400]."""
        assert self.model.window_sizes == [20, 100, 400, 2400]

    def test_per_lead_processing(self) -> None:
        """7 leads should produce 7 output feature vectors."""
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape[1] == 7

    def test_gradient_flow(self) -> None:
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

    def test_output_dim_property(self) -> None:
        """output_dim should reflect channel doubling through patch merging."""
        # 3 downsamples → embed_dim * 2^3 = 128 * 8 = 1024
        assert self.model.output_dim == 1024
