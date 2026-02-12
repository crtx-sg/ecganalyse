"""U-Net ECG denoiser with residual learning."""

from __future__ import annotations

import torch
import torch.nn as nn


class _EncoderBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool1d(2)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class _DecoderBlock(nn.Module):
    """ConvTranspose1d (upsample) → concat skip → Conv1d → BN → ReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle length mismatch from non-power-of-2 inputs
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, (0, diff))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ECGDenoiser(nn.Module):
    """U-Net denoiser with residual learning.

    Predicts noise from the input and subtracts it:
        output = input - predicted_noise

    Args:
        in_channels: Number of ECG leads (default: 7).
        base_channels: Number of channels in the first encoder block (default: 32).
        depth: Number of encoder/decoder stages (default: 4).
    """

    def __init__(
        self,
        in_channels: int = 7,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth

        # Encoder
        enc_channels = [in_channels] + [base_channels * (2**i) for i in range(depth)]
        self.encoders = nn.ModuleList(
            _EncoderBlock(enc_channels[i], enc_channels[i + 1])
            for i in range(depth)
        )

        # Bottleneck
        bot_ch = enc_channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bot_ch, bot_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(bot_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(bot_ch * 2, bot_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(bot_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            dec_in = enc_channels[i + 1]
            skip_ch = enc_channels[i + 1]
            dec_out = enc_channels[i] if i > 0 else base_channels
            self.decoders.append(_DecoderBlock(dec_in, skip_ch, dec_out))

        # Final 1×1 conv to predict noise (same channels as input)
        self.final = nn.Conv1d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, 7, 2400].

        Returns:
            Denoised tensor of shape [batch, 7, 2400].
        """
        identity = x

        # Encoder path
        skips: list[torch.Tensor] = []
        h = x
        for enc in self.encoders:
            h, skip = enc(h)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder path (skips in reverse order)
        for dec, skip in zip(self.decoders, reversed(skips)):
            h = dec(h, skip)

        # Predict noise
        noise = self.final(h)

        # Handle length mismatch (from pooling/unpooling rounding)
        if noise.shape[-1] != identity.shape[-1]:
            noise = noise[..., : identity.shape[-1]]

        # Residual: output = input - noise
        return identity - noise
