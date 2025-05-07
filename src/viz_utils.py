"""Visualization utilities for graphs and node embeddings."""
from typing import Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_graph(G: nx.Graph, color: Union[str, list]):
    """
    Visualize the graph using NetworkX and Matplotlib.
    Args:
        G (nx.Graph): Graph to visualize.
        color (Union[str, list]): Color for the nodes.
    """
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=42),
        with_labels=False,
        node_color=color,
        cmap="Set2",
    )
    plt.show()


def visualize_embedding(
    h: np.array,
    color: Union[str, list],
    epoch: Optional[Union[int, float]] = None,
    loss: Optional[np.float32] = None,
):
    """
    Visualize the node embeddings in 2D space.
    Args:
        h (np.array): Node embeddings.
        color (Union[str, list]): Color for the nodes.
        epoch (Union[int,float], optional): Epoch number. Defaults to None.
        loss (Union[int,float], optional): Loss value. Defaults to None.
    """
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(
            f"Epoch: {epoch}, Loss: {loss.item(): .4f}", fontsize=16
        )  # noqa: E231
    plt.show()
