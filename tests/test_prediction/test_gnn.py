"""Unit tests for LeadGNN."""

import torch
import pytest

from src.prediction.gnn import LeadGNN, LEAD_ADJACENCY, LEAD_NAMES


class TestLeadGNN:

    def setup_method(self) -> None:
        self.model = LeadGNN(d_in=256, d_hidden=128, n_layers=2, n_heads=4)
        self.model.eval()

    def test_graph_structure(self) -> None:
        """Graph should have 7 nodes."""
        assert self.model.num_nodes == 7

    def test_edge_count(self) -> None:
        """9 undirected edges (18 directed) + 7 self-loops = 25."""
        assert self.model.num_edges == 25

    def test_anatomical_adjacency(self) -> None:
        """Should include the specified anatomical edges."""
        expected = {
            ("ECG1", "ECG2"), ("ECG1", "aVL"), ("ECG2", "ECG3"),
            ("ECG2", "aVF"), ("ECG3", "aVF"), ("aVR", "aVL"),
            ("aVR", "aVF"), ("vVX", "ECG1"), ("vVX", "ECG2"),
        }
        actual = {(a, b) for a, b in LEAD_ADJACENCY}
        assert actual == expected

    def test_output_shapes(self) -> None:
        """graph_embedding [B, d_hidden], node_embeddings [B, 7, d_hidden]."""
        x = torch.randn(2, 7, 256)
        with torch.no_grad():
            g_emb, n_emb = self.model(x)
        assert g_emb.shape == (2, 128)
        assert n_emb.shape == (2, 7, 128)

    def test_gradient_flow(self) -> None:
        self.model.train()
        x = torch.randn(1, 7, 256)
        g_emb, n_emb = self.model(x)
        loss = g_emb.sum() + n_emb.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
        )
        assert has_grad

    def test_attention_weights_learnable(self) -> None:
        """GAT attention parameters should be learnable."""
        for layer in self.model.layers:
            assert layer.a_l.requires_grad
            assert layer.a_r.requires_grad

    def test_batch_dimension(self) -> None:
        x = torch.randn(4, 7, 256)
        with torch.no_grad():
            g_emb, n_emb = self.model(x)
        assert g_emb.shape[0] == 4

    def test_finite_output(self) -> None:
        x = torch.randn(1, 7, 256)
        with torch.no_grad():
            g_emb, n_emb = self.model(x)
        assert torch.isfinite(g_emb).all()
        assert torch.isfinite(n_emb).all()
