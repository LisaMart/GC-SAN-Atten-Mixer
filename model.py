#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light-weight session-based recommender without GNN.

Core idea
---------
1. Embed each item and normalize with L2 norm to ensure unit length.
2. Apply multi-head self-attention over session items to capture relationships.
3. Fuse last hidden state with attention-pooled vector to compute session representation.
4. Mask already-seen items during inference to prevent recommending items already interacted.
5. Optimize cross-entropy loss combined with attention diversity regularizer.

Key classes
-----------
PositionEmbedding              – adds positional information to item embeddings (add/concat/expand modes).
SimpleMultiHeadAttention      – wrapper around torch.nn.MultiheadAttention with attention storage for regularization.
SessionGraphWithMultiLevelAttention – main model class.
train_test                    – performs one epoch of training and evaluation.
"""

import datetime
import math
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from tqdm import tqdm

# -------------------- 1. Positional Embedding --------------------
class PositionEmbedding(nn.Module):
    """
    Adds positional information to embeddings using three modes:
    - MODE_ADD: element-wise addition of positional embedding
    - MODE_CONCAT: concatenates positional embedding to original embedding
    - MODE_EXPAND: index-based expanded embedding lookup
    """
    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD    = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self, num_embeddings: int, embedding_dim: int, mode: str = MODE_ADD):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.mode = mode
        size = num_embeddings * 2 + 1 if mode == self.MODE_EXPAND else num_embeddings
        self.weight = nn.Parameter(torch.empty(size, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using Xavier initialization
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute positional embeddings according to mode
        if self.mode == self.MODE_EXPAND:
            idx = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(idx.long(), self.weight)

        batch_size, seq_len = x.shape[:2]
        pos = self.weight[:seq_len].view(1, seq_len, self.embedding_dim)

        if self.mode == self.MODE_ADD:
            return x + pos
        if self.mode == self.MODE_CONCAT:
            return torch.cat([x, pos.expand(batch_size, -1, -1)], dim=-1)
        raise NotImplementedError(f"Unknown mode {self.mode}")

# -------------------- 2. Residual Block --------------------
class Residual(Module):
    """
    Simple residual block with two linear layers and dropout.
    Adds input to the output to stabilize training and enable deeper networks.
    """
    def __init__(self, hidden_size: int = 120, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(self.dp(x))
        return residual + x

# -------------------- 3. Multi-head Attention Wrapper --------------------
class SimpleMultiHeadAttention(nn.Module):
    """
    Wraps torch.nn.MultiheadAttention and stores attention weights for regularization.
    Inverts mask because PyTorch uses 1 = keep, 0 = pad.
    """
    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=heads, batch_first=True)

    def forward(self, query, key, value, mask=None):
        # Convert mask to PyTorch format: True = ignore, False = attend
        key_padding_mask = (~mask.bool()) if mask is not None else None
        out, weights = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        self.alpha = weights  # store attention weights for diversity regularizer
        return out, weights

# -------------------- 4. Main Session Model --------------------
class SessionGraphWithMultiLevelAttention(Module):
    """
    Implements session-based recommendation model using multi-level attention.
    Embeds items, applies self-attention, pools attention, and predicts next item.
    """
    def __init__(self, opt, n_node: int, len_max: int):
        super().__init__()
        self.hidden_size = opt.hiddenSize
        self.embedding = nn.Embedding(n_node, self.hidden_size, padding_idx=0)

        # Multi-head attention to capture intra-session relations
        self.multi_level_attn = SimpleMultiHeadAttention(self.hidden_size, opt.heads)

        # Linear transformation to fuse pooled attention and last hidden state
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        # Loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Uniform initialization for all model parameters
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

    # ---------------- forward pass ----------------
    def forward(self, inputs: torch.Tensor, A, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute session hidden representations.
        inputs: [B, L] item indices
        A: None (placeholder for compatibility with other models)
        mask: [B, L], 1 for real items, 0 for padding
        returns: hidden states after self-attention
        """
        hidden = self.embedding(inputs)          # convert indices to embeddings [B, L, H]
        hidden = F.normalize(hidden, p=2, dim=-1) # L2 normalize embeddings

        # Apply multi-level self-attention
        hidden, _ = self.multi_level_attn(hidden, hidden, hidden, mask)
        return hidden

    # ---------------- score computation ----------------
    def compute_scores(self, hidden, inputs, mask, mask_seen=False):
        idx = mask.sum(1) - 1
        idx = torch.clamp(idx, 0, hidden.size(1) - 1)
        ht = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    
        mask_f = mask.float().unsqueeze(-1)
        attn_out = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
    
        out = self.linear_transform(torch.cat([attn_out, ht], dim=-1))
        scores = out @ self.embedding.weight[1:].T
    
        if mask_seen:
            for b in range(hidden.size(0)):
                seen = inputs[b].masked_select(mask[b].bool())
                scores[b, seen - 1] = -1e9
    
        return scores

    # ---------------- loss with diversity regularizer ----------------
    def loss_function(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.loss_fn(scores, targets)                     # standard cross-entropy
        attn_std = torch.std(self.multi_level_attn.alpha, dim=-1).mean()  # attention diversity
        return ce - 0.001 * attn_std

# -------------------- 5. Device helpers --------------------
def trans_to_cuda(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to GPU if available"""
    return tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def trans_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor back to CPU"""
    return tensor.cpu()

# -------------------- 6. Forward for one batch --------------------
def forward(model: SessionGraphWithMultiLevelAttention, idx: int, data) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract one batch from dataset, run forward, compute scores"""
    inputs, _, _, mask, targets = data.get_slice(idx)
    inputs  = trans_to_cuda(torch.tensor(inputs, dtype=torch.long))
    mask    = trans_to_cuda(torch.tensor(mask, dtype=torch.long))
    targets = trans_to_cuda(torch.tensor(targets, dtype=torch.long))

    hidden = model(inputs, None, mask)
    scores = model.compute_scores(hidden, inputs, mask, mask_seen=False)
    return targets, scores

# -------------------- 7. Train + evaluate one epoch --------------------
def train_test(model, train_data, test_data, epoch, batch_size):
    print('start training:', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(batch_size)

    for i in tqdm(slices, desc=f"Training Epoch {epoch}"):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

    model.scheduler.step()

    avg_loss = total_loss / len(slices)
    print(f'\tTotal Loss:\t{total_loss:.4f}\n\tAverage Loss:\t{avg_loss:.4f}')

    print('start predicting:', datetime.datetime.now())
    model.eval()

    top_K = [5, 10, 20]
    metrics = {K: {"precision": [], "mrr": []} for K in top_K}

    with torch.no_grad():
        slices = test_data.generate_batch(batch_size)

        for i in tqdm(slices, desc="Testing"):
            targets, scores = forward(model, i, test_data)
            targets = targets - 1

            max_k = max(top_K)
            top_items = scores.topk(max_k, dim=1).indices

            for K in top_K:
                pred = top_items[:, :K]
                hit_matrix = pred.eq(targets.view(-1, 1))
                hit = hit_matrix.any(dim=1)

                metrics[K]["precision"].append(hit.float().mean().item() * 100)

                ranks = hit_matrix.float().argmax(dim=1) + 1
                rr = torch.where(
                    hit,
                    1.0 / ranks.float(),
                    torch.zeros_like(ranks, dtype=torch.float)
                )
                metrics[K]["mrr"].append(rr.mean().item() * 100)

    result = {"loss": avg_loss}
    for K in top_K:
        result[K] = {
            "precision": float(np.mean(metrics[K]["precision"])),
            "mrr": float(np.mean(metrics[K]["mrr"]))
        }

    return result
