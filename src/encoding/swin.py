"""1D Swin Transformer encoder for ECG morphological feature extraction.

Hierarchical shifted-window attention with 4 levels optimised for 200 Hz ECG:
  Level 1: window 20 samples  (100 ms — wave morphology)
  Level 2: window 100 samples (500 ms — single beat)
  Level 3: window 400 samples (2 s   — beat-to-beat)
  Level 4: window 2400 samples (12 s — global rhythm)

Input : [batch, 7, 2400]
Output: [batch, 7, seq_len, embed_dim]
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1-D windowed self-attention
# ---------------------------------------------------------------------------

class _WindowAttention1D(nn.Module):
    """Multi-head self-attention within fixed-size windows."""

    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Learnable relative position bias
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * window_size - 1)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [num_windows * B, window_size, dim]."""
        NW, W, D = x.shape
        qkv = self.qkv(x).reshape(NW, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, NW, H, W, Hd]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [NW, H, W, W]

        # Add relative position bias
        pos_idx = torch.arange(W, device=x.device)
        rel_idx = pos_idx.unsqueeze(0) - pos_idx.unsqueeze(1) + (self.window_size - 1)
        rel_idx = rel_idx.clamp(0, 2 * self.window_size - 2)
        bias = self.rel_pos_bias[:, rel_idx]  # [H, W, W]
        attn = attn + bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(NW, W, D)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Swin Transformer block (window attn + shifted window attn + FFN)
# ---------------------------------------------------------------------------

class _SwinBlock1D(nn.Module):
    """One Swin Transformer block: W-MSA → FFN → SW-MSA → FFN."""

    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention1D(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.norm3 = nn.LayerNorm(dim)
        self.shifted_attn = _WindowAttention1D(dim, num_heads, window_size)
        self.norm4 = nn.LayerNorm(dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D]."""
        B, L, D = x.shape
        ws = self.window_size

        # --- Regular window attention ---
        h = self.norm1(x)
        h = self._window_partition_attn(h, B, L, ws, self.attn)
        x = x + h
        x = x + self.ffn(self.norm2(x))

        # --- Shifted window attention ---
        shift = ws // 2
        h = self.norm3(x)
        h = torch.roll(h, shifts=-shift, dims=1)
        h = self._window_partition_attn(h, B, L, ws, self.shifted_attn)
        h = torch.roll(h, shifts=shift, dims=1)
        x = x + h
        x = x + self.ffn2(self.norm4(x))

        return x

    @staticmethod
    def _window_partition_attn(
        x: torch.Tensor, B: int, L: int, ws: int, attn_fn: nn.Module,
    ) -> torch.Tensor:
        """Partition into windows, apply attention, merge."""
        # Pad to multiple of window_size
        pad = (ws - L % ws) % ws
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        L_padded = L + pad
        num_windows = L_padded // ws

        # Partition: [B, num_win, ws, D]
        x = x.reshape(B, num_windows, ws, -1)
        x = x.reshape(B * num_windows, ws, -1)

        # Attention
        x = attn_fn(x)

        # Merge
        x = x.reshape(B, num_windows, ws, -1).reshape(B, L_padded, -1)
        if pad > 0:
            x = x[:, :L, :]
        return x


# ---------------------------------------------------------------------------
# Patch merging (downsamples sequence length by 2x, doubles channels)
# ---------------------------------------------------------------------------

class _PatchMerging1D(nn.Module):
    """Merge adjacent patches: [B, L, D] → [B, L//2, 2*D]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(dim * 2, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        # Pad if odd length
        if L % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            L = L + 1
        x0 = x[:, 0::2, :]  # even
        x1 = x[:, 1::2, :]  # odd
        x = torch.cat([x0, x1], dim=-1)  # [B, L//2, 2D]
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ---------------------------------------------------------------------------
# Full Swin1D stage (blocks + optional downsampling)
# ---------------------------------------------------------------------------

class _SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _SwinBlock1D(dim, num_heads, window_size) for _ in range(depth)
        )
        self.downsample = _PatchMerging1D(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class Swin1DTransformer(nn.Module):
    """Hierarchical 1D Swin Transformer for ECG feature extraction.

    Args:
        embed_dim:    Base embedding dimension (default 128).
        depths:       Number of blocks per stage (default [2,2,6,2]).
        num_heads:    Attention heads per stage (default [4,8,16,32]).
        window_sizes: Window sizes per stage in samples (default [20,100,400,2400]).
        patch_size:   Initial patch tokenisation size (default 4).
        in_channels:  Per-lead input channels (default 1).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (4, 8, 16, 32),
        window_sizes: Sequence[int] = (20, 100, 400, 2400),
        patch_size: int = 4,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_stages = len(depths)
        self.window_sizes = list(window_sizes)

        # Patch embedding
        self.patch_embed = nn.Linear(patch_size * in_channels, embed_dim)
        self.pos_drop = nn.Dropout(0.0)

        # Build stages
        self.stages = nn.ModuleList()
        dim = embed_dim
        for i in range(self.num_stages):
            # Window size in token units
            ws_tokens = max(1, window_sizes[i] // patch_size // (2 ** i))
            downsample = i < self.num_stages - 1
            self.stages.append(
                _SwinStage(dim, depths[i], num_heads[i], ws_tokens, downsample)
            )
            if downsample:
                dim = dim * 2  # _PatchMerging doubles channels

        self.norm = nn.LayerNorm(dim)
        self._output_dim = dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[batch, 7, 2400]``.

        Returns:
            Feature tensor of shape ``[batch, 7, seq_len, output_dim]``.
        """
        B, num_leads, T = x.shape
        seq_len = T // self.patch_size

        lead_outputs = []
        for lead_idx in range(num_leads):
            lead = x[:, lead_idx, :]                             # [B, T]
            patches = lead.reshape(B, seq_len, self.patch_size)  # [B, S, P]
            tokens = self.patch_embed(patches)                   # [B, S, embed_dim]
            tokens = self.pos_drop(tokens)

            for stage in self.stages:
                tokens = stage(tokens)
            tokens = self.norm(tokens)                           # [B, S', D']
            lead_outputs.append(tokens)

        return torch.stack(lead_outputs, dim=1)  # [B, 7, S', D']
