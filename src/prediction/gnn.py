"""Graph Attention Network for multi-lead ECG fusion.

Pure-PyTorch GAT implementation (no ``torch_geometric`` dependency).
Models spatial relationships between the 7 ECG leads using an
anatomically-informed adjacency graph.

Input : [batch, 7, d_features]
Output: graph_embedding [batch, d_hidden], node_embeddings [batch, 7, d_hidden]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Anatomically-informed lead adjacency
LEAD_NAMES = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]
LEAD_INDEX = {name: i for i, name in enumerate(LEAD_NAMES)}

LEAD_ADJACENCY = [
    ("ECG1", "ECG2"),
    ("ECG1", "aVL"),
    ("ECG2", "ECG3"),
    ("ECG2", "aVF"),
    ("ECG3", "aVF"),
    ("aVR", "aVL"),
    ("aVR", "aVF"),
    ("vVX", "ECG1"),
    ("vVX", "ECG2"),
]


def _build_edge_index() -> torch.Tensor:
    """Build edge index tensor from adjacency list (undirected → both directions).

    Returns:
        ``[2, num_edges]`` long tensor.
    """
    edges: list[tuple[int, int]] = []
    for a, b in LEAD_ADJACENCY:
        i, j = LEAD_INDEX[a], LEAD_INDEX[b]
        edges.append((i, j))
        edges.append((j, i))
    # Add self-loops
    for k in range(len(LEAD_NAMES)):
        edges.append((k, k))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class _GATLayer(nn.Module):
    """Single multi-head graph attention layer."""

    def __init__(self, d_in: int, d_out: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        assert d_out % n_heads == 0

        self.W = nn.Linear(d_in, d_out, bias=False)
        # Attention parameters: a_l, a_r for each head
        self.a_l = nn.Parameter(torch.empty(n_heads, self.d_head))
        self.a_r = nn.Parameter(torch.empty(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.a_l)
        nn.init.xavier_uniform_(self.a_r)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_in]  (N = 7 nodes).
            edge_index: [2, E] edge indices.

        Returns:
            [B, N, d_out].
        """
        B, N, _ = x.shape
        residual = x

        h = self.W(x)                                # [B, N, d_out]
        h = h.view(B, N, self.n_heads, self.d_head)  # [B, N, H, Dh]

        src, dst = edge_index[0], edge_index[1]       # [E]

        # Attention coefficients
        # a_l dot h[src] + a_r dot h[dst]
        h_src = h[:, src]                              # [B, E, H, Dh]
        h_dst = h[:, dst]                              # [B, E, H, Dh]

        e = (h_src * self.a_l).sum(-1) + (h_dst * self.a_r).sum(-1)  # [B, E, H]
        e = self.leaky_relu(e)

        # Softmax per destination node
        # Build attention weights: for each dst node, softmax over incoming edges
        attn = torch.full((B, N, self.n_heads), float("-inf"), device=x.device)
        # We need to scatter e into the correct positions
        # For each edge (src→dst), the attention from src to dst
        # We aggregate messages at dst
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.n_heads)
        attn.scatter_(1, dst_expanded, e)

        # Actually we need proper sparse softmax. Use a simpler approach:
        # For each node, gather its neighbor attention scores and softmax
        out = torch.zeros(B, N, self.n_heads, self.d_head, device=x.device)
        for node in range(N):
            # Find edges where dst == node
            mask = dst == node
            if not mask.any():
                continue
            edge_attn = e[:, mask]            # [B, num_neighbors, H]
            edge_attn = F.softmax(edge_attn, dim=1)
            edge_attn = self.dropout(edge_attn)
            neighbor_h = h_src[:, mask]       # [B, num_neighbors, H, Dh]
            # Weighted sum
            weighted = (edge_attn.unsqueeze(-1) * neighbor_h).sum(1)  # [B, H, Dh]
            out[:, node] = weighted

        out = out.reshape(B, N, -1)            # [B, N, d_out]

        # Residual + norm (project residual if dim mismatch)
        if residual.shape[-1] != out.shape[-1]:
            out = self.norm(out)
        else:
            out = self.norm(out + residual)

        return F.elu(out)


class LeadGNN(nn.Module):
    """Graph Attention Network for multi-lead ECG fusion.

    Args:
        d_in:      Input feature dimension per node.
        d_hidden:  Hidden dimension (default 128).
        n_layers:  Number of GAT layers (default 2).
        n_heads:   Number of attention heads (default 4).
        dropout:   Dropout rate (default 0.1).
    """

    def __init__(
        self,
        d_in: int = 256,
        d_hidden: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden

        # Input projection
        self.input_proj = nn.Linear(d_in, d_hidden)

        # GAT layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(_GATLayer(d_hidden, d_hidden, n_heads, dropout))

        # Graph-level readout
        self.graph_proj = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Register edge index as buffer (not a parameter)
        self.register_buffer("edge_index", _build_edge_index())

    @property
    def num_nodes(self) -> int:
        return len(LEAD_NAMES)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def forward(
        self,
        node_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            node_features: ``[batch, 7, d_in]`` per-lead features.

        Returns:
            Tuple of:
                - ``graph_embedding``: ``[batch, d_hidden]`` — graph-level summary.
                - ``node_embeddings``: ``[batch, 7, d_hidden]`` — enhanced per-lead features.
        """
        h = self.input_proj(node_features)  # [B, 7, d_hidden]

        for layer in self.layers:
            h = layer(h, self.edge_index)   # [B, 7, d_hidden]

        node_embeddings = h

        # Graph-level readout: mean pooling + projection
        graph_pool = h.mean(dim=1)          # [B, d_hidden]
        graph_embedding = self.graph_proj(graph_pool)  # [B, d_hidden]

        return graph_embedding, node_embeddings
