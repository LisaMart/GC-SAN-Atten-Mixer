#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main training and evaluation script for LightGCL + Atten-Mixer.

This version keeps the previous data formulation:
seq[0] is treated as user/session id,
seq[1:] are treated as interacted items.

Metrics:
Precision@5, Precision@10, Precision@20,
MRR@5, MRR@10, MRR@20.
"""

import os
import re
import warnings
from typing import List

import numpy as np
import torch
import torch.utils.data as data
from scipy.sparse import coo_matrix
from tqdm import tqdm

from model import LightGCL
from parser import args
from utils import TrnData, scipy_sparse_mat_to_torch_sparse_tensor


warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------- device setup --------------------
if args.cuda == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

print(f"device: {device}")


# -------------------- hyper-parameters --------------------
d = args.d
l = args.gnn_layer
temp = args.temp

batch_u = args.batch
inter_b = args.inter_batch

epoch_no = args.epoch
lr = args.lr

lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout

svd_q = args.q


# -------------------- dataset path --------------------
base_path = os.path.join(os.path.dirname(__file__), "datasets")
ds_path = os.path.join(base_path, args.data)

train_path = os.path.join(ds_path, "train.txt")
test_path = os.path.join(ds_path, "test.txt")

print("train path:", train_path)
print("test path:", test_path)


# -------------------- 1. Load sessions --------------------
def load_sessions(path: str) -> List[List[int]]:
    """
    Read a session file into a list of integer sequences.
    Each non-empty line becomes one sequence.
    """
    for enc in ("utf-8", "utf-16", "latin1"):
        try:
            sessions = []

            with open(path, encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    seq = [int(x) for x in re.findall(r"\d+", line)]
                    seq = [x for x in seq if x != 0]

                    if len(seq) >= 2:
                        sessions.append(seq)

            return sessions

        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"cannot read {path}")


# -------------------- 2. Remap IDs --------------------
def remap_sessions(session_list: List[List[int]]):
    """
    Map raw user/session ids and item ids to dense indices.

    Convention:
    seq[0] -> user/session id
    seq[1:] -> item ids
    """
    session_list = [s for s in session_list if len(s) >= 2]

    user_set = {s[0] for s in session_list}

    item_set = set()
    for s in session_list:
        item_set.update(s[1:])

    umap = {u: idx for idx, u in enumerate(sorted(user_set))}
    imap = {i: idx for idx, i in enumerate(sorted(item_set))}

    remapped = []

    for s in session_list:
        uid = umap[s[0]]
        items = [imap[i] for i in s[1:] if i in imap]

        if len(items) > 0:
            remapped.append([uid] + items)

    return remapped, umap, imap


def remap_test_sessions(session_list: List[List[int]], umap, imap):
    """
    Remap test sessions using train mappings.
    Unknown users/items are removed.
    """
    remapped = []

    for s in session_list:
        if len(s) < 2:
            continue

        uid = umap.get(s[0], -1)
        if uid < 0:
            continue

        items = [imap[i] for i in s[1:] if i in imap]

        if len(items) > 0:
            remapped.append([uid] + items)

    return remapped


# -------------------- 3. Build COO matrix --------------------
def build_coo(sessions: List[List[int]]) -> coo_matrix:
    """
    Convert sessions into a user-item COO sparse matrix.

    Convention:
    seq[0] -> user/session id
    seq[1:] -> interacted items
    """
    if not sessions:
        return coo_matrix((0, 0), dtype=np.float32)

    rows, cols, vals = [], [], []

    for seq in sessions:
        if len(seq) < 2:
            continue

        uid = seq[0]

        for iid in seq[1:]:
            rows.append(uid)
            cols.append(iid)
            vals.append(1.0)

    if not rows:
        return coo_matrix((0, 0), dtype=np.float32)

    return coo_matrix(
        (vals, (rows, cols)),
        shape=(max(rows) + 1, max(cols) + 1),
        dtype=np.float32
    )


# -------------------- 4. Evaluation --------------------
def evaluate_model(model, test_labels, batch_u, device):
    """
    Compute Precision@5/10/20 and MRR@5/10/20.
    Values are returned in percent.
    """
    model.eval()

    eval_uids = np.array(
        [uid for uid, labels in enumerate(test_labels) if len(labels) > 0],
        dtype=np.int64
    )

    if len(eval_uids) == 0:
        return {
            "P@5": 0.0,
            "P@10": 0.0,
            "P@20": 0.0,
            "MRR@5": 0.0,
            "MRR@10": 0.0,
            "MRR@20": 0.0,
            "eval_users": 0,
        }

    ks = [5, 10, 20]
    max_k = max(ks)

    hits = {k: 0.0 for k in ks}
    mrr_sum = {k: 0.0 for k in ks}

    batches = (len(eval_uids) + batch_u - 1) // batch_u

    with torch.no_grad():
        for b in range(batches):
            start = b * batch_u
            end = min((b + 1) * batch_u, len(eval_uids))

            batch_uids_np = eval_uids[start:end]
            batch_uids = torch.LongTensor(batch_uids_np).to(device)

            preds = model(
                batch_uids,
                None,
                None,
                None,
                test=True,
                topk=max_k
            ).cpu().numpy()

            for row_idx, uid in enumerate(batch_uids_np):
                labels = set(test_labels[uid])

                for k in ks:
                    topk_items = preds[row_idx][:k].tolist()

                    hits[k] += float(len(set(topk_items) & labels) > 0)

                    rr = 0.0
                    for rank, item in enumerate(topk_items, start=1):
                        if item in labels:
                            rr = 1.0 / rank
                            break

                    mrr_sum[k] += rr

    n_eval = len(eval_uids)

    return {
        "P@5": hits[5] / n_eval * 100,
        "P@10": hits[10] / n_eval * 100,
        "P@20": hits[20] / n_eval * 100,
        "MRR@5": mrr_sum[5] / n_eval * 100,
        "MRR@10": mrr_sum[10] / n_eval * 100,
        "MRR@20": mrr_sum[20] / n_eval * 100,
        "eval_users": n_eval,
    }

# -------------------- 5. Load & preprocess --------------------
train_seq_raw = load_sessions(train_path)
test_seq_raw = load_sessions(test_path)

print(f"loaded {len(train_seq_raw)} train / {len(test_seq_raw)} test sessions")

train_seq, umap, imap = remap_sessions(train_seq_raw)
test_seq = remap_test_sessions(test_seq_raw, umap, imap)

print("after remap/filter:")
print("train sessions:", len(train_seq))
print("test sessions:", len(test_seq))
print("unique train users:", len(umap))
print("unique train items:", len(imap))


# -------------------- 6. Interaction matrices --------------------
train_mat = build_coo(train_seq)
test_mat = build_coo(test_seq)

print("train shape:", train_mat.shape, "nnz:", train_mat.nnz)
print("test  shape:", test_mat.shape, "nnz:", test_mat.nnz)

if train_mat.nnz == 0:
    raise RuntimeError("train_mat is empty. Check train.txt format and remapping.")

train_csr = (train_mat != 0).astype(np.float32)

n_u, n_i = train_mat.shape

print("n_u:", n_u)
print("n_i:", n_i)


# -------------------- 7. Test labels --------------------
test_labels = [[] for _ in range(n_u)]

test_mat_coo = test_mat.tocoo()

for r, c in zip(test_mat_coo.row, test_mat_coo.col):
    if r < n_u and c < n_i:
        test_labels[r].append(c)

eval_users_count = sum(1 for labels in test_labels if len(labels) > 0)

print("eval users:", eval_users_count)

if eval_users_count == 0:
    print("WARNING: no evaluation users. Metrics will be zero.")


# -------------------- 8. Symmetric normalization --------------------
train_coo = train_mat.tocoo().astype(np.float32)

row_sum = np.array(train_coo.sum(1)).squeeze()
col_sum = np.array(train_coo.sum(0)).squeeze()

for i in range(len(train_coo.data)):
    r = train_coo.row[i]
    c = train_coo.col[i]

    denom = np.sqrt(row_sum[r] * col_sum[c])
    if denom > 0:
        train_coo.data[i] /= denom

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train_coo).coalesce().to(device)


# -------------------- 9. SVD factors --------------------
svd_u, s, svd_v = torch.svd_lowrank(adj_norm, q=svd_q)

u_mul_s = svd_u @ torch.diag(s)
v_mul_s = svd_v @ torch.diag(s)

del s


# -------------------- 10. Model & optimizer --------------------
model = LightGCL(
    n_u,
    n_i,
    d,
    u_mul_s,
    v_mul_s,
    svd_u.T,
    svd_v.T,
    train_csr,
    adj_norm,
    l,
    temp,
    lambda_1,
    lambda_2,
    dropout,
    batch_u,
    device,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epoch_no,
    eta_min=1e-5
)


# -------------------- 11. Training loop --------------------
best = {
    "P@5": 0.0,
    "P@10": 0.0,
    "P@20": 0.0,
    "MRR@5": 0.0,
    "MRR@10": 0.0,
    "MRR@20": 0.0,
}

best_epoch = {key: 0 for key in best}

train_loader = data.DataLoader(
    TrnData(train_mat),
    batch_size=inter_b,
    shuffle=True,
    num_workers=4
)

for epoch in range(epoch_no):
    train_loader.dataset.neg_sampling()

    model.train()

    total_loss = 0.0
    total_bpr_loss = 0.0
    total_ssl_loss = 0.0

    for uids, pos, neg in tqdm(train_loader, desc=f"epoch {epoch}"):
        uids = uids.long().to(device)
        pos = pos.long().to(device)
        neg = neg.long().to(device)

        optimizer.zero_grad()

        loss, loss_r, loss_s = model(
            uids,
            torch.cat([pos, neg]),
            pos,
            neg
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bpr_loss += loss_r.item()
        total_ssl_loss += loss_s.item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_bpr_loss = total_bpr_loss / len(train_loader)
    avg_ssl_loss = total_ssl_loss / len(train_loader)

    print(
        f"epoch {epoch:03d} | "
        f"loss={avg_loss:.4f} | "
        f"bpr={avg_bpr_loss:.4f} | "
        f"ssl={avg_ssl_loss:.4f}"
    )

    metrics = evaluate_model(model, test_labels, batch_u, device)

    print(
        f"epoch {epoch:03d} | "
        f"P@5={metrics['P@5']:.4f} | "
        f"MRR@5={metrics['MRR@5']:.4f} | "
        f"P@10={metrics['P@10']:.4f} | "
        f"MRR@10={metrics['MRR@10']:.4f} | "
        f"P@20={metrics['P@20']:.4f} | "
        f"MRR@20={metrics['MRR@20']:.4f} | "
        f"eval_users={metrics['eval_users']}"
    )

    for key in best:
        if metrics[key] > best[key]:
            best[key] = metrics[key]
            best_epoch[key] = epoch

    print(
        "Best so far | "
        f"P@5={best['P@5']:.4f}({best_epoch['P@5']}) | "
        f"MRR@5={best['MRR@5']:.4f}({best_epoch['MRR@5']}) | "
        f"P@10={best['P@10']:.4f}({best_epoch['P@10']}) | "
        f"MRR@10={best['MRR@10']:.4f}({best_epoch['MRR@10']}) | "
        f"P@20={best['P@20']:.4f}({best_epoch['P@20']}) | "
        f"MRR@20={best['MRR@20']:.4f}({best_epoch['MRR@20']})"
    )
