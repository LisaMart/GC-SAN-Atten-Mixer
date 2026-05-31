# -------------------- Updated main.py with detailed English comments --------------------

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
from typing import List, Dict, Tuple
import time
import torch.multiprocessing as mp
from utils import build_graph, Data, split_validation
from model import trans_to_cuda, SessionGraphWithMultiLevelAttention, train_test
import os

# Ensure compatibility on Windows
mp.set_start_method('spawn', force=True)

# -------------------- 1. I/O helpers --------------------
def load_sessions_from_txt(path: str) -> List[List[int]]:
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


def remap_sessions(sessions: List[List[int]], existing_map: Dict[int, int] = None) -> Tuple[List[List[int]], Dict[int, int]]:
    """Remap original item IDs to contiguous indices starting from 1 (0 reserved for padding).
    
    Args:
        sessions: list of item sequences
        existing_map: if provided, reuse this mapping (for test set).
                     New items not in existing_map get new IDs.
    
    Returns:
        remapped sessions and the (updated) item_map dictionary
    """
    if existing_map is not None:
        item_map = existing_map.copy()
    else:
        item_map = {}
    
    next_id = max(item_map.values(), default=0) + 1
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

def build_examples(sessions):
    xs, ys = [], []
    for s in sessions:
        s = [x for x in s if x != 0]
        if len(s) >= 2:
            xs.append(s[:-1])
            ys.append(s[-1])
    return xs, ys

# -------------------- 2. CLI argument parser --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--heads', type=int, default=4, help='number of attention heads')
    p.add_argument('--dataset', default='yoochoose1_64', help='dataset folder under datasets/')
    p.add_argument('--batchSize', type=int, default=50)
    p.add_argument('--hiddenSize', type=int, default=120)
    p.add_argument('--epoch', type=int, default=30)
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
    
    project_root = "/home/lisa/GC-SAN+Atten-Mixer"
    base_path = os.path.join(project_root, "datasets")
    
    train_path = os.path.join(base_path, opt.dataset, "train.txt")
    test_path = os.path.join(base_path, opt.dataset, "test.txt")
    
    train_raw = load_sessions_from_txt(train_path)
    test_raw = load_sessions_from_txt(test_path)
    
    print("Train path:", train_path)
    print("Test path:", test_path)
    
    train_raw = [s for s in train_raw if len(s) >= 2 and s[-1] != 0]
    test_raw = [s for s in test_raw if len(s) >= 2 and s[-1] != 0]
    
    all_sessions = train_raw + test_raw
    all_sessions, item_map = remap_sessions(all_sessions)
    
    train_sessions = all_sessions[:len(train_raw)]
    test_sessions = all_sessions[len(train_raw):]
    
    train_x, train_y = build_examples(train_sessions)
    test_x, test_y = build_examples(test_sessions)
    
    n_node = len(item_map) + 1
    print(f"Automatically inferred n_node = {n_node}")
    print(f"Train examples: {len(train_x)}")
    print(f"Test examples: {len(test_x)}")
    
    train_data = Data((train_x, train_y), shuffle=True, opt=opt)
    test_data = Data((test_x, test_y), shuffle=False, opt=opt)

    # Optional validation split
    if opt.validation:
        train_sessions, valid_sessions = split_validation(
            (train_sessions, [s[-1] for s in train_sessions]), opt.valid_portion
        )
        test_sessions = valid_sessions[0]
        print(f'After split – train: {len(train_sessions)}, valid: {len(test_sessions)}')


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
