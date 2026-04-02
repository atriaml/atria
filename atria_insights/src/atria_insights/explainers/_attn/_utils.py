from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def attention_adjacency_matrix(attentions: torch.Tensor):
    """Construct a layered adjacency matrix from attention weights.

    Args:
        attentions: Tensor of shape (n_layers, seq_len, seq_len) containing attention weights.

    Returns:
        A numpy array of shape (total_nodes, total_nodes) representing the layered adjacency matrix.
    """
    n_layers, length, _ = attentions.shape
    total_nodes = (n_layers + 1) * length
    adjacency_matrix = np.zeros((total_nodes, total_nodes), dtype=float)

    for layer in range(1, n_layers + 1):
        row_start = layer * length
        col_start = (layer - 1) * length
        adjacency_matrix[
            row_start : row_start + length, col_start : col_start + length
        ] = attentions[layer - 1]

    return adjacency_matrix


def compute_flows(attentions: torch.Tensor) -> np.ndarray:
    """Compute normalized max-flow values from each layer node to input nodes using igraph.

    Args:
        attentions: Tensor of shape (n_layers, seq_len, seq_len) containing attention weights.

    Returns:
        A numpy array of shape (total_nodes, total_nodes) containing normalized flow values.
    """
    import igraph as ig

    n_layers, sequence_length, _ = attentions.shape
    total_nodes = (n_layers + 1) * sequence_length

    adj_mat = attention_adjacency_matrix(attentions)

    # Build igraph from non-zero edges only
    rows, cols = np.nonzero(adj_mat)
    edges = list(zip(rows.tolist(), cols.tolist(), strict=True))
    capacities = adj_mat[rows, cols].tolist()

    graph = ig.Graph(n=total_nodes, edges=edges, directed=True)
    graph.es["capacity"] = capacities

    input_node_indices = range(sequence_length)
    flow_values = np.zeros((total_nodes, total_nodes), dtype=float)

    for node_idx in range(sequence_length, total_nodes):
        current_layer = node_idx // sequence_length
        previous_layer = current_layer - 1

        for input_node_idx in input_node_indices:
            flow_value = graph.maxflow_value(
                node_idx, input_node_idx, capacity="capacity"
            )
            flow_values[node_idx, previous_layer * sequence_length + input_node_idx] = (
                flow_value
            )

        row_sum = flow_values[node_idx].sum()
        if row_sum > 0:
            flow_values[node_idx] /= row_sum

    return adjacency_to_layerwise(
        flow_values, n_layers=attentions.shape[0], n_tokens=attentions.shape[-1]
    )


def adjacency_to_layerwise(
    adjacency_matrix: np.ndarray, n_layers: int, n_tokens: int
) -> np.ndarray:
    """Recover per-layer attention matrices from a stacked adjacency matrix."""
    layer_wise_score_matrix = np.zeros((n_layers, n_tokens, n_tokens), dtype=float)

    for layer in range(n_layers):
        layer_wise_score_matrix[layer] = adjacency_matrix[
            (layer + 1) * n_tokens : (layer + 2) * n_tokens,
            layer * n_tokens : (layer + 1) * n_tokens,
        ]

    return layer_wise_score_matrix
