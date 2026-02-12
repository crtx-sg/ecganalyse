"""Unit tests for DualPathFusion and FoundationModelAdapter."""

import torch
import pytest

from src.encoding.fusion import DualPathFusion
from src.encoding.foundation import FoundationModelAdapter


class TestDualPathFusion:

    def test_concat_project_shape(self) -> None:
        """Fusion produces [B, 7, S_fused, output_dim]."""
        fusion = DualPathFusion(mamba_dim=256, swin_dim=1024, output_dim=256)
        fusion.eval()

        m = torch.randn(2, 7, 240, 256)
        s = torch.randn(2, 7, 75, 1024)
        with torch.no_grad():
            out = fusion(m, s)
        assert out.shape == (2, 7, 240, 256)

    def test_same_seq_len(self) -> None:
        """When both inputs have same seq_len, no interpolation needed."""
        fusion = DualPathFusion(mamba_dim=128, swin_dim=128, output_dim=64)
        fusion.eval()
        m = torch.randn(1, 7, 100, 128)
        s = torch.randn(1, 7, 100, 128)
        with torch.no_grad():
            out = fusion(m, s)
        assert out.shape == (1, 7, 100, 64)

    def test_deterministic_shape(self) -> None:
        """Output shape is deterministic for given config."""
        fusion = DualPathFusion(mamba_dim=256, swin_dim=1024, output_dim=256)
        fusion.eval()
        m = torch.randn(1, 7, 240, 256)
        s = torch.randn(1, 7, 75, 1024)
        with torch.no_grad():
            out1 = fusion(m, s)
            out2 = fusion(m, s)
        assert out1.shape == out2.shape

    def test_gradient_flow(self) -> None:
        fusion = DualPathFusion(mamba_dim=256, swin_dim=1024, output_dim=256)
        fusion.train()
        m = torch.randn(1, 7, 240, 256)
        s = torch.randn(1, 7, 75, 1024)
        out = fusion(m, s)
        loss = out.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in fusion.parameters()
        )
        assert has_grad

    def test_finite_output(self) -> None:
        fusion = DualPathFusion(mamba_dim=256, swin_dim=1024, output_dim=256)
        fusion.eval()
        m = torch.randn(1, 7, 240, 256)
        s = torch.randn(1, 7, 75, 1024)
        with torch.no_grad():
            out = fusion(m, s)
        assert torch.isfinite(out).all()


class TestFoundationModelAdapter:

    def test_dual_path_fallback(self) -> None:
        """Without foundation model, uses Mamba + Swin + Fusion."""
        adapter = FoundationModelAdapter(output_dim=256)
        assert not adapter.uses_foundation
        adapter.eval()
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = adapter(x)
        assert y.shape == (1, 7, 240, 256)

    def test_foundation_model_used(self) -> None:
        """With foundation model provided, delegates to it."""
        class DummyFoundation(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B = x.shape[0]
                return torch.randn(B, 7, 100, 256)

        adapter = FoundationModelAdapter(
            foundation_model=DummyFoundation(), output_dim=256,
        )
        assert adapter.uses_foundation
        adapter.eval()
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = adapter(x)
        assert y.shape == (1, 7, 100, 256)

    def test_no_error_random_weights(self) -> None:
        """Dual-path with random weights should not error."""
        adapter = FoundationModelAdapter(output_dim=256)
        adapter.eval()
        x = torch.randn(2, 7, 2400)
        with torch.no_grad():
            y = adapter(x)
        assert torch.isfinite(y).all()
