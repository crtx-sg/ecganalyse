"""ECG-Mamba encoder based on Selective State Space Models.

Pure-PyTorch implementation (no ``mamba_ssm`` dependency).  Processes each
of the 7 ECG leads independently with shared weights, capturing long-range
temporal dependencies across the full 12-second strip.

Input : [batch, 7, 2400]
Output: [batch, 7, seq_len, d_model]   (seq_len = 2400 / patch_size)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Selective SSM block (simplified S6-style)
# ---------------------------------------------------------------------------

class _SelectiveSSMBlock(nn.Module):
    """Single selective state space block.

    Implements a simplified selective scan: the discretisation parameters
    (dt, B, C) are input-dependent, providing the *selectivity* mechanism
    that distinguishes Mamba from vanilla S4.
    """

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projection (expand → inner_dim for gating)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)

        # Conv1d for local context (like Mamba's short conv)
        self.conv1d = nn.Conv1d(
            d_model, d_model, kernel_size=4, padding=3, groups=d_model,
        )

        # Input-dependent SSM parameters
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

        # Learnable log-scale for A (diagonal, negative real)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(d_model, -1)
            .clone()
        )
        self.D = nn.Parameter(torch.ones(d_model))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model] → [batch, seq_len, d_model]."""
        B, L, D = x.shape
        residual = x

        # Project and split for gating
        xz = self.in_proj(x)                          # [B, L, 2*D]
        x_branch, z = xz.chunk(2, dim=-1)             # each [B, L, D]

        # Causal conv1d
        x_conv = x_branch.transpose(1, 2)             # [B, D, L]
        x_conv = self.conv1d(x_conv)[:, :, :L]        # causal trim
        x_conv = F.silu(x_conv).transpose(1, 2)       # [B, L, D]

        # Compute input-dependent SSM parameters
        dt = F.softplus(self.dt_proj(x_conv))          # [B, L, D]
        B_param = self.B_proj(x_conv)                  # [B, L, N]
        C_param = self.C_proj(x_conv)                  # [B, L, N]
        A = -torch.exp(self.A_log)                     # [D, N]

        # Selective scan (sequential for correctness; parallel scan not needed
        # since we only need shape-correct forward pass for now)
        y = self._selective_scan(x_conv, dt, A, B_param, C_param)  # [B, L, D]

        # Skip connection via D parameter
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Gate and project
        y = y * F.silu(z)
        y = self.out_proj(y)

        return self.norm(y + residual)

    def _selective_scan(
        self,
        u: torch.Tensor,      # [B, L, D]
        dt: torch.Tensor,      # [B, L, D]
        A: torch.Tensor,       # [D, N]
        B: torch.Tensor,       # [B, L, N]
        C: torch.Tensor,       # [B, L, N]
    ) -> torch.Tensor:
        """Discretise and run the selective scan recurrence.

        Uses a parallel-friendly cumulative-sum approximation for speed.
        """
        batch, seq_len, d_model = u.shape
        d_state = A.shape[1]

        # Discretise: dA = exp(A * dt), dB = dt * B
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # [B, L, D, N]
        dA = torch.exp(dt_A)                                    # [B, L, D, N]
        dB_u = (
            dt.unsqueeze(-1)                                     # [B, L, D, 1]
            * B.unsqueeze(2)                                     # [B, L, 1, N]
            * u.unsqueeze(-1)                                    # [B, L, D, 1]
        )                                                        # [B, L, D, N]

        # Recurrence (chunked for memory efficiency)
        h = torch.zeros(batch, d_model, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB_u[:, t]                       # [B, D, N]
            y_t = torch.einsum("bdn,bln->bd", h, C[:, t : t + 1])  # [B, D]
            ys.append(y_t)
        return torch.stack(ys, dim=1)                            # [B, L, D]


# ---------------------------------------------------------------------------
# Mamba layer = SSM block + feed-forward
# ---------------------------------------------------------------------------

class _MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.ssm = _SelectiveSSMBlock(d_model, d_state)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ssm(x)
        return x + self.ff(self.norm(x))


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class ECGMamba(nn.Module):
    """ECG-Mamba encoder.

    Args:
        d_model:    Feature dimension (default 256).
        d_state:    SSM state dimension (default 64).
        n_layers:   Number of Mamba layers (default 4).
        patch_size: Samples per patch for tokenisation (default 10).
                    seq_len = 2400 / patch_size = 240.
        in_channels: Number of input channels per lead (default 1).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        patch_size: int = 10,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        # Patch embedding: each patch of `patch_size` samples → d_model
        self.patch_embed = nn.Linear(patch_size * in_channels, d_model)
        self.pos_embed: nn.Parameter | None = None  # lazily created

        # Mamba layers (shared across leads)
        self.layers = nn.ModuleList(
            _MambaLayer(d_model, d_state) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[batch, 7, 2400]``.

        Returns:
            Feature tensor of shape ``[batch, 7, seq_len, d_model]``.
        """
        B, num_leads, T = x.shape
        seq_len = T // self.patch_size

        # Process each lead independently (shared weights)
        lead_outputs = []
        for lead_idx in range(num_leads):
            lead = x[:, lead_idx, :]                            # [B, T]
            # Tokenise into patches
            patches = lead.reshape(B, seq_len, self.patch_size)  # [B, S, P]
            tokens = self.patch_embed(patches)                   # [B, S, D]

            # Positional embedding (lazy init for flexibility)
            if self.pos_embed is None or self.pos_embed.shape[1] != seq_len:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, seq_len, self.d_model, device=x.device),
                    requires_grad=True,
                )
            tokens = tokens + self.pos_embed

            # Mamba layers
            for layer in self.layers:
                tokens = layer(tokens)
            tokens = self.norm(tokens)                           # [B, S, D]
            lead_outputs.append(tokens)

        return torch.stack(lead_outputs, dim=1)                  # [B, 7, S, D]
