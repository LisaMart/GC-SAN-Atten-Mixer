#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from tqdm import tqdm

# ---------- Position Embedding ----------
class PositionEmbedding(nn.Module):
    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD    = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self, num_embeddings, embedding_dim, mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.mode = mode
        self.weight = nn.Parameter(
            torch.Tensor(num_embeddings * 2 + 1 if mode == self.MODE_EXPAND else num_embeddings,
                         embedding_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.long(), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError

# ---------- Residual block ----------
class Residual(Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 120
        self.d1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.d2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dp = nn.Dropout(p=0.2)

    def forward(self, x):
        residual = x
        x = F.relu(self.d1(x))
        x = self.d2(self.dp(x))
        return residual + x

# ---------- Simple Multi-Head Attention ----------
class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=heads, batch_first=True)

    def forward(self, query, key, value, mask=None):
        key_padding_mask = (~mask.bool()) if mask is not None else None
        out, weights = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        self.alpha = weights
        return out, weights

# ---------- Main Model ----------
class SessionGraphWithMultiLevelAttention(Module):
    def __init__(self, opt, n_node, len_max):
        super().__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node      = n_node
        self.len_max     = len_max
        self.embedding   = nn.Embedding(self.n_node, self.hidden_size)

        # attention
        self.multi_level_attn = SimpleMultiHeadAttention(self.hidden_size, opt.heads)

        # prediction
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        # optimization
        self.loss_fn   = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

    # ---------- ablation: w/o attention & residual ----------
    def compute_scores(self, hidden, inputs, mask):
        """
        hidden : [B, L, H]
        inputs : [B, L]   (raw item ids)
        mask   : [B, L]   (1/0)
        """
        # last item embedding
        idx = torch.sum(mask, dim=1) - 1
        idx = torch.clamp(idx, 0, hidden.size(1) - 1)
        ht = hidden[torch.arange(hidden.size(0)), idx]          # [B, H]

        # dot-product with all candidate items (skip id=0 padding)
        scores = torch.matmul(ht, self.embedding.weight[1:].t())  # [B, n_node-1]

        # mask already-seen items
        for b in range(hidden.size(0)):
            seen = inputs[b].masked_select(mask[b].bool())      # items in this session
            scores[b, seen - 1] = -1e9                         # 0-based after skip

        return scores

    def forward(self, inputs, A, mask):
        # simple embedding
        hidden = self.embedding(inputs)                       # [B, L, H]
        hidden = F.normalize(hidden, p=2, dim=-1)
        return hidden

    # Изменение функции потерь
    def loss_function(self, scores, targets):
        # Перекрёстная энтропия для классификационной задачи - Базовая функция потерь
        loss = torch.nn.CrossEntropyLoss()(scores, targets)
        attn_std = torch.std(self.multi_level_attn.alpha, dim=-1).mean()
        # Отнимаем небольшую величину — поощрение разнообразного распределения
        loss -= 0.001 * attn_std  # мягкое поощрение разнообразия

        return loss

    def forward(self, inputs, A, mask):
        hidden = self.embedding(inputs)  # Применяем embedding к входным данным
        hidden = F.normalize(hidden, p=2, dim=-1)

        # Проверка размерности и добавление оси времени, если необходимо
        if len(hidden.size()) == 2:  # Если тензор двухмерный, добавляем ось времени
            hidden = hidden.unsqueeze(1)  # Добавляем ось seq_len: [batch_size, 1, hidden_size]

        # Применяем многослойное внимание
        attn_output, _ = self.multi_level_attn(query=hidden, key=hidden, value=hidden, mask=mask)

        # Получаем последнее скрытое состояние
        idx = torch.sum(mask, dim=1) - 1
        idx = torch.clamp(idx, min=0, max=hidden.size(1) - 1)
        ht = hidden[torch.arange(hidden.size(0)).long(), idx]

        # Преобразуем ht в 2D тензор (если он 1D) для объединения
        if ht.dim() == 1:
            ht = ht.unsqueeze(1)  # Преобразуем в [batch_size, hidden_size]

        attn_output, _ = self.multi_level_attn(query=hidden, key=hidden, value=hidden, mask=mask)
        attn_output = attn_output.mean(dim=1)  # Сжимаем по временной оси
        out = self.linear_transform(torch.cat([attn_output, ht], dim=-1))  # Объединяем внимание и скрытое состояние
        return hidden

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        #return variable.cuda()
        return variable.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def forward(model, i, data):
    inputs, _, _, mask, targets = data.get_slice(i)

    # Переводим в тензоры и на GPU
    inputs = trans_to_cuda(torch.tensor(inputs).long())         # [batch_size, seq_len]
    mask = trans_to_cuda(torch.tensor(mask).long())             # [batch_size, seq_len]
    targets = trans_to_cuda(torch.tensor(targets).long())       # [batch_size]

    # Получаем скрытые состояния (без GNN)
    hidden = model(inputs, None, mask)                          # [batch_size, seq_len, hidden_size]

    # Индексы последних активных элементов
    idx = torch.sum(mask, dim=1) - 1                            # [batch_size]
    idx = torch.clamp(idx, min=0, max=hidden.size(1) - 1)       # гарантируем границы

    # Собираем скрытые состояния с помощью индексирования
    seq_hidden = hidden                                         # [batch_size, seq_len, hidden_size]

    return targets, model.compute_scores_ablation(seq_hidden, mask)

def train_test(model, train_data, test_data, epoch, batch_size):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(batch_size)      # <-- исправлено

    for i in tqdm(slices, desc=f"Training Epoch {epoch}"):
        model.optimizer.zero_grad()
        inputs, _, _, mask, targets = train_data.get_slice(i)
        inputs  = trans_to_cuda(torch.tensor(inputs).long())
        mask    = trans_to_cuda(torch.tensor(mask).long())
        targets = trans_to_cuda(torch.tensor(targets).long())

        hidden = model(inputs, None, mask)
        scores = model.compute_scores(hidden, inputs, mask)

        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        model.scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(slices)
    print('\tTotal Loss:\t%.4f\n\tAverage Loss:\t%.4f' % (total_loss, avg_loss))

    # ---- evaluation ----
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    top_K = [5, 10, 20]
    metrics = {K: {"precision": [], "mrr": []} for K in top_K}
    slices = test_data.generate_batch(batch_size)

    for i in tqdm(slices, desc="Testing"):
        inputs, _, _, mask, targets = test_data.get_slice(i)
        inputs  = trans_to_cuda(torch.tensor(inputs).long())
        mask    = trans_to_cuda(torch.tensor(mask).long())
        targets = trans_to_cuda(torch.tensor(targets).long())

        hidden = model(inputs, None, mask)
        scores = model.compute_scores(hidden, inputs, mask)

        scores  = trans_to_cpu(scores).detach().numpy()
        targets = trans_to_cpu(targets).long().numpy() - 1

        for K in top_K:
            top_indices = scores.argsort(axis=1)[:, -K:][:, ::-1]
            for pred, label in zip(top_indices, targets):
                hit = np.isin(label, pred)
                metrics[K]["precision"].append(float(hit))
                if hit:
                    rank = np.where(pred == label)[0][0] + 1
                    metrics[K]["mrr"].append(1.0 / rank)
                else:
                    metrics[K]["mrr"].append(0.0)

    return {
        "loss": avg_loss,
        **{K: {"precision": np.mean(metrics[K]["precision"]) * 100,
               "mrr": np.mean(metrics[K]["mrr"]) * 100} for K in top_K}
    }