"""Unit tests for ECGDenoiser U-Net."""

import torch
import pytest

from src.preprocessing.denoiser import ECGDenoiser


class TestECGDenoiser:

    def setup_method(self) -> None:
        self.model = ECGDenoiser(in_channels=7, base_channels=32, depth=4)
        self.model.eval()

    def test_output_shape(self) -> None:
        """Forward pass: [1,7,2400] → [1,7,2400]."""
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape == (1, 7, 2400), f"Expected (1,7,2400), got {y.shape}"

    def test_output_dtype(self) -> None:
        """Output dtype should be float32."""
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        assert y.dtype == torch.float32

    def test_residual_structure(self) -> None:
        """The model uses residual learning: output = input - noise.
        With random weights the noise won't be zero, but the subtraction
        structure should mean the output is different from (but related to) input.
        """
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        # Output should not be identical to input (noise prediction is non-zero)
        assert not torch.allclose(x, y, atol=1e-6)
        # But it should be finite
        assert torch.isfinite(y).all()

    def test_energy_preservation(self) -> None:
        """Output RMS should be within 50% of input RMS (not zeroed out)."""
        x = torch.randn(1, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        rms_in = torch.sqrt(torch.mean(x**2)).item()
        rms_out = torch.sqrt(torch.mean(y**2)).item()
        # Output energy should be at least 50% of input
        assert rms_out > 0.5 * rms_in, (
            f"RMS ratio too low: {rms_out:.4f} vs {rms_in:.4f}"
        )

    def test_batch_dimension(self) -> None:
        """Batch size > 1 should work."""
        x = torch.randn(4, 7, 2400)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape == (4, 7, 2400)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the model (trainability check)."""
        x = torch.randn(1, 7, 2400, requires_grad=False)
        y = self.model(x)
        loss = y.sum()
        loss.backward()
        # At least some parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
        )
        assert has_grad, "No gradients detected"
