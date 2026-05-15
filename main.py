#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end training script for GC-SAN with Multi-Level Attention.

Pipeline
--------
1. Load session sequences from text files.
2. Remove padding items (item 0).
3. Remap original item IDs to dense indices (0 reserved for padding).
4. Create Data objects for PyTorch training/testing.
5. Train and validate with early stopping.
6. Compute and track metrics: Precision@K and MRR@K.
7. Print best metrics and epoch achieved.
"""

import argparse
import re
import time
import torch.multiprocessing as mp
from utils import build_graph, Data, split_validation
from model import trans_to_cuda, SessionGraphWithMultiLevelAttention, train_test
import os

# Ensure compatibility on Windows
mp.set_start_method('spawn', force=True)

# -------------------- 1. I/O helpers --------------------
def load_sessions_from_txt(path: str) -> list[list[int]]:
    """Read text file where each line represents a session; return list of item sequences."""
    sessions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            numbers = re.findall(r'\d+', line)
            item_seq = list(map(int, numbers))
            if item_seq:
                sessions.append(item_seq)
    return sessions


def remap_sessions(sessions: list[list[int]]) -> tuple[list[list[int]], dict[int, int]]:
    """Remap original item IDs to contiguous indices starting from 1 (0 reserved for padding)."""
    item_map, next_id = {}, 1
    remapped = []
    for seq in sessions:
        new_seq = []
        for item in seq:
            if item not in item_map:
                item_map[item] = next_id
                next_id += 1
            new_seq.append(item_map[item])
        remapped.append(new_seq)
    return remapped, item_map

# -------------------- 2. CLI argument parser --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--heads', type=int, default=4, help='number of attention heads')
    p.add_argument('--dataset', default='yoochoose1_64', help='dataset folder under datasets/')
    p.add_argument('--batchSize', type=int, default=50)
    p.add_argument('--hiddenSize', type=int, default=120)
    p.add_argument('--epoch', type=int, default=1)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay factor')
    p.add_argument('--lr_dc_step', type=int, default=3, help='decay every N epochs')
    p.add_argument('--l2', type=float, default=1e-5, help='weight decay')
    p.add_argument('--patience', type=int, default=10, help='early stopping patience')
    p.add_argument('--nonhybrid', action='store_true', help='disable local + global hybrid')
    p.add_argument('--validation', action='store_true', help='enable train/validation split')
    p.add_argument('--valid_portion', type=float, default=0.1)
    p.add_argument('--dynamic', type=bool, default=False, help='dynamic label flag (unused)')
    p.add_argument('--dot', type=float, default=0.1, help='hyperparameter for Atten-Mixer')
    p.add_argument('--l_p', type=float, default=1.0, help='hyperparameter for Atten-Mixer')
    p.add_argument('--last_k', type=int, default=3, help='number of last clicks used in Atten-Mixer')
    p.add_argument('--use_attn_conv', action='store_true', help='enable attentional convolution')
    return p.parse_args()

# -------------------- 3. Main function --------------------
def main():
    opt = parse_args()
    print(opt)

    # Load training and test sessions
    base_path = os.path.join(os.path.dirname(__file__), 'datasets')
    train_sessions = load_sessions_from_txt(f'{base_path}\\{opt.dataset}\\train.txt')
    test_sessions = load_sessions_from_txt(f'{base_path}\\{opt.dataset}\\test.txt')

    # Remove sessions ending with padding item 0
    train_sessions = [s for s in train_sessions if s[-1] != 0]
    test_sessions = [s for s in test_sessions if s[-1] != 0]

    # Remap item IDs to contiguous indices
    train_sessions, _ = remap_sessions(train_sessions)
    test_sessions, _ = remap_sessions(test_sessions)

    # Determine number of unique nodes (0 = padding)
    all_items = [item for seq in train_sessions + test_sessions for item in seq]
    n_node = max(all_items) + 1
    print(f'Automatically inferred n_node = {n_node}')

    # Optional validation split
    if opt.validation:
        train_sessions, valid_sessions = split_validation(
            (train_sessions, [s[-1] for s in train_sessions]), opt.valid_portion
        )
        test_sessions = valid_sessions[0]
        print(f'After split – train: {len(train_sessions)}, valid: {len(test_sessions)}')

    # Build Data objects for training and testing
    train_data = Data((train_sessions, [s[-1] for s in train_sessions]), shuffle=True, opt=opt)
    test_data = Data((test_sessions, [s[-1] for s in test_sessions]), shuffle=False, opt=opt)

    # Instantiate model and move to GPU if available
    model = trans_to_cuda(
        SessionGraphWithMultiLevelAttention(
            opt, n_node, max(train_data.len_max, test_data.len_max)
        )
    )

    # Training loop with early stopping
    start = time.time()
    best_result = {K: [0.0, 0.0] for K in [5, 10, 20]}  # store best P@K and MRR@K
    best_epoch = {K: [0, 0] for K in [5, 10, 20]}        # epoch where best metrics achieved
    best_loss = float('inf')
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print(f'epoch: {epoch}/{opt.epoch - 1}')
        metrics = train_test(model, train_data, test_data, epoch, opt.batchSize)

        # Track best metrics
        improved = 0
        for K in [5, 10, 20]:
            if metrics[K]['precision'] > best_result[K][0]:
                best_result[K][0] = metrics[K]['precision']
                best_epoch[K][0] = epoch
                improved = 1
            if metrics[K]['mrr'] > best_result[K][1]:
                best_result[K][1] = metrics[K]['mrr']
                best_epoch[K][1] = epoch
                improved = 1
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            improved = 1

        # Pretty print best metrics
        print('Best Result:')
        for K in [5, 10, 20]:
            print(f'  P@{K}: {best_result[K][0]:.4f}  MRR@{K}: {best_result[K][1]:.4f}  Epoch: {best_epoch[K]}')
        print(f'  Best Loss: {best_loss:.4f}')

        # Early stopping
        bad_counter += 1 - improved
        if bad_counter >= opt.patience:
            print(f'No improvement for {opt.patience} epochs – stopping.')
            break

    print('-------------------------------------------------------')
    print(f'Run time: {time.time() - start:.2f} s')

if __name__ == '__main__':
    main()
