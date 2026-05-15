#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility helpers for session-based recommendation.

Functions
---------
build_graph       – builds a weighted directed graph capturing transition probabilities
data_masks        – pads sequences to uniform length and generates masks
split_validation  – splits dataset into train/validation sets randomly

Class
-----
Data – minimal Dataset class compatible with PyTorch, returns batches of padded sequences
"""

from typing import List, Tuple, Optional
import networkx as nx
import numpy as np

# -------------------- 1. Build weighted directed graph --------------------
def build_graph(train_data: List[List[int]]) -> nx.DiGraph:
    """
    Create a directed graph capturing item transitions.
    Edge weight = number of times transition occurs.
    We normalize incoming edges to sum to 1 (probability-like).
    """
    graph = nx.DiGraph()

    # Count all transitions in sequences
    for seq in train_data:
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            if graph.has_edge(u, v):
                graph[u][v]["weight"] += 1
            else:
                graph.add_edge(u, v, weight=1)

    # Normalize incoming edges per node
    for node in graph.nodes:
        in_weight_sum = sum(graph[j][node]["weight"] for j, _ in graph.in_edges(node))
        if in_weight_sum:
            for j, _ in graph.in_edges(node):
                graph[j][node]["weight"] /= in_weight_sum
    return graph

# -------------------- 2. Pad sequences and generate masks --------------------
def data_masks(all_usr_pois: List[List[int]], item_tail: int) -> Tuple[List[List[int]], List[List[int]], int]:
    """
    Pad sequences to same length with item_tail (typically 0).
    Returns padded sequences, masks, and max length.
    """
    us_lens = [len(seq) for seq in all_usr_pois]
    len_max = max(us_lens) if us_lens else 0

    padded_seqs = [seq + [item_tail] * (len_max - le) for seq, le in zip(all_usr_pois, us_lens)]
    masks = [[1] * le + [0] * (len_max - le) for le in us_lens]

    return padded_seqs, masks, len_max

# -------------------- 3. Random train/validation split --------------------
def split_validation(train_set: Tuple[List, List], valid_portion: float) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    """
    Randomly split a dataset into training and validation subsets.
    train_set: tuple(inputs, labels)
    valid_portion: fraction to allocate to validation
    Returns ((train_x, train_y), (valid_x, valid_y))
    """
    train_x, train_y = train_set
    n_samples = len(train_x)
    idx = np.arange(n_samples, dtype="int32")
    np.random.shuffle(idx)

    n_train = int(n_samples * (1 - valid_portion))
    train_x = [train_x[i] for i in idx[:n_train]]
    train_y = [train_y[i] for i in idx[:n_train]]
    valid_x = [train_x[i] for i in idx[n_train:]]
    valid_y = [train_y[i] for i in idx[n_train:]]
    return (train_x, train_y), (valid_x, valid_y)

# -------------------- 4. Minimal Dataset class --------------------
class Data:
    """
    Minimal PyTorch-compatible dataset class.
    Does NOT build adjacency matrices. Returns batches ready for training/testing.
    """
    def __init__(self, data: Tuple[List[List[int]], List[int]], shuffle: bool = False, graph: Optional[nx.DiGraph] = None, opt=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, item_tail=0)

        self.inputs = np.asarray(inputs)      # padded input sequences
        self.mask = np.asarray(mask)          # 1 for real items, 0 for padding
        self.len_max = len_max
        self.targets = np.asarray(data[1])    # next-item targets
        if opt and getattr(opt, 'dynamic', False):
            self.targets = np.asarray(data[2])  # alternative dynamic labels if specified
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size: int) -> List[np.ndarray]:
        """Return a list of mini-batch indices."""
        if self.shuffle:
            idx = np.arange(self.length)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx]
            self.mask = self.mask[idx]
            self.targets = self.targets[idx]

        n_batch = (self.length + batch_size - 1) // batch_size
        slices = np.array_split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:self.length - batch_size * (n_batch - 1)]  # trim last slice
        return slices

    def get_slice(self, idx: np.ndarray) -> Tuple:
        """Return minimal slice for a batch, no adjacency matrices built."""
        return self.inputs[idx], None, None, self.mask[idx], self.targets[idx]
