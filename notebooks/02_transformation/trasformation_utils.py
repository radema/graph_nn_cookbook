import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data, InMemoryDataset


def StoreDataset(data_list: List[Data], folder: str, filename: str = "dataset"):
    """
    Store a PyTorch Geometric dataset to disk.

    Args:
        dataset (InMemoryDataset): The dataset to store.
        path (str): The path to store the dataset.
    """
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), folder + "/" + filename + ".pt")


class LoadedDataset(InMemoryDataset):
    """
    Load a PyTorch Geometric dataset from disk.

    Args:
        path (str): The path to load the dataset from.
    """

    def __init__(self, folder: str, filename: str = "dataset"):
        super(LoadedDataset, self).__init__(folder)
        self.data, self.slices = torch.load(folder + "/" + filename + ".pt")
        self.data = self.data.to(self.__class__.__device__)
        self.slices = {
            key: val.to(self.__class__.__device__) for key, val in self.slices.items()
        }


def create_pyg_graph_from_polars(
    df: pl.DataFrame, or_col: str, dest_col: str, edge_attr_cols: list
):
    """
    Build a PyG Data graph from columns of a Polars DataFrame.

    Parameters:
        df: polars.DataFrame
        or_col: str - Name of the origin node column
        dest_col: str - Name of the destination node column
        edge_attr_cols: list of str - List of column names for edge features

    Returns:
        PyG Data object with attributes: edge_index, edge_attr, num_nodes, node2idx, idx2node
    """
    # 1. Unique node labels and mapping
    all_nodes = np.unique(
        np.concatenate([df[or_col].to_numpy(), df[dest_col].to_numpy()])
    )
    node2idx = {code: idx for idx, code in enumerate(all_nodes)}

    # 2. Build edge_index
    src_idx = [node2idx[x] for x in df[or_col].to_numpy()]
    dest_idx = [node2idx[x] for x in df[dest_col].to_numpy()]
    edge_index = torch.tensor([src_idx, dest_idx], dtype=torch.long)

    # 3. Edge attributes
    edge_attr = torch.tensor(
        np.column_stack([df[col].to_numpy() for col in edge_attr_cols]),
        dtype=torch.float,
    )

    # 4. Create Data object
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(all_nodes))
    data.node2idx = node2idx
    data.idx2node = {idx: code for code, idx in node2idx.items()}

    return data
