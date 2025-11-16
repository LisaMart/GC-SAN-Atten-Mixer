#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import re
import time
import torch.multiprocessing as mp
from utils import build_graph, Data, split_validation
from model import *

mp.set_start_method('spawn')

# ---------- загрузка ----------
def load_sessions_from_txt(path):
    sessions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            numbers = re.findall(r'\d+', line)
            item_seq = list(map(int, numbers))
            if item_seq:                    # пропускаем пустые
                sessions.append(item_seq)
    return sessions

# ---------- перенумерация (remap) ----------
def remap_sessions(sessions):
    item_map = {}
    next_id = 1                     # 0 оставляем для PAD
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

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heads', type=int, default=4,
                        help='number of attention heads')
    parser.add_argument('--dataset', default='yoochoose1_64',
                        help='dataset name: diginetica/sample/yoochoose1_64')
    parser.add_argument('--batchSize', type=int, default=50)
    parser.add_argument('--hiddenSize', type=int, default=120)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_dc', type=float, default=0.1)
    parser.add_argument('--lr_dc_step', type=int, default=3)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--nonhybrid', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--valid_portion', type=float, default=0.1)
    parser.add_argument('--dynamic', type=bool, default=False)
    parser.add_argument('--dot', type=float, default=0.1)
    parser.add_argument('--l_p', type=float, default=1)
    parser.add_argument('--last_k', type=int, default=3)
    parser.add_argument('--use_attn_conv', action='store_true')
    opt = parser.parse_args()
    print(opt)

    # 1. Загрузка сессий
    base_path = r'C:\Users\Chaingun\Desktop\lisa\01_GC-SAN_am-gnn\datasets'
    train_sessions = load_sessions_from_txt(f'{base_path}\\{opt.dataset}\\train.txt')
    test_sessions  = load_sessions_from_txt(f'{base_path}\\{opt.dataset}\\test.txt')

    # (опционально) быстрое уменьшение объёма
    # train_sessions = train_sessions[:2000]
    # test_sessions  = test_sessions[:500]

    # 2. Фильтрация
    train_sessions = [s for s in train_sessions if s[-1] != 0]
    test_sessions  = [s for s in test_sessions  if s[-1] != 0]

    # 3. Перенумерация айтемов
    train_sessions, _ = remap_sessions(train_sessions)
    test_sessions,  _ = remap_sessions(test_sessions)

    # 4. Автоматический n_node
    all_items = [item for seq in train_sessions + test_sessions for item in seq]
    n_node = max(all_items) + 1
    print(f'Автоматически определён n_node = {n_node}')

    # 5. Validation split (если нужен)
    if opt.validation:
        train_sessions, valid_sessions = split_validation(
            (train_sessions, [s[-1] for s in train_sessions]),
            opt.valid_portion
        )
        test_sessions = valid_sessions[0]
        print(f'После split_validation: train={len(train_sessions)}, valid={len(test_sessions)}')

    # 6. Подготовка Data-объектов
    train_data = Data((train_sessions, [s[-1] for s in train_sessions]),
                      shuffle=True, opt=opt)
    test_data  = Data((test_sessions,  [s[-1] for s in test_sessions]),
                      shuffle=False, opt=opt)

    # 7. Модель
    model = trans_to_cuda(
        SessionGraphWithMultiLevelAttention(
            opt, n_node, max(train_data.len_max, test_data.len_max)
        )
    )

    # ---------- обучение ----------
    start = time.time()
    best_result = {K: [0.0, 0.0] for K in [5, 10, 20]}
    best_epoch  = {K: [0, 0] for K in [5, 10, 20]}
    best_loss   = float('inf')
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print(f'epoch: {epoch}/{opt.epoch - 1}')
        #metrics = train_test(model, train_data, test_data, epoch)
        metrics = train_test(model, train_data, test_data, epoch, opt.batchSize)

        flag = 0
        for K in [5, 10, 20]:
            if metrics[K]['precision'] > best_result[K][0]:
                best_result[K][0] = metrics[K]['precision']
                best_epoch[K][0]  = epoch
                flag = 1
            if metrics[K]['mrr'] > best_result[K][1]:
                best_result[K][1] = metrics[K]['mrr']
                best_epoch[K][1]  = epoch
                flag = 1

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            flag = 1

        print('Best Result:')
        for K in [5, 10, 20]:
            print(f'  P@{K}: {best_result[K][0]:.4f}  MRR@{K}: {best_result[K][1]:.4f}  Epoch: {best_epoch[K]}')
        print(f'  Best Loss: {best_loss:.4f}')

        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break

    print('-------------------------------------------------------')
    end = time.time()
    print(f'Run time: {end - start:.2f} s')

if __name__ == '__main__':
    main()