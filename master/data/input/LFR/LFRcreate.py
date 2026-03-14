# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp


def to_dense_float32(x):
    """把 mat 里的矩阵转成 numpy.float32 dense"""
    if sp.issparse(x):
        x = x.toarray()
    else:
        x = np.asarray(x)
    return x.astype(np.float32)


def extract_time_slice(mat_path, time_index=7):
    """
    从 LFR-data10x.mat 中提取指定时间片
    time_index: 按自然编号传入，例如 7 表示第7个时间片
    """
    data = sio.loadmat(mat_path)

    if "W_Cube" not in data or "GT_Cube" not in data:
        raise KeyError(f"{mat_path} 中缺少 W_Cube 或 GT_Cube")

    W_cube = data["W_Cube"]   # shape: (1, 10), object
    GT_cube = data["GT_Cube"] # shape: (1, 10), object

    t = int(time_index) - 1   # Python 索引
    if t < 0 or t >= W_cube.shape[1]:
        raise ValueError(f"time_index={time_index} 超出范围，当前共有 {W_cube.shape[1]} 个时间片")

    W_t = W_cube[0, t]
    GT_t = GT_cube[0, t]

    A = to_dense_float32(W_t)
    GT = to_dense_float32(GT_t)

    # 基本检查
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{mat_path} 的 W_t 不是方阵，shape={A.shape}")

    if GT.ndim != 2 or GT.shape[0] != A.shape[0]:
        raise ValueError(f"{mat_path} 的 GT_t 维度不匹配，A.shape={A.shape}, GT.shape={GT.shape}")

    # 对称化 + 去自环（保险起见）
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    # one-hot -> label
    row_sums = GT.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"{mat_path} 的 GT_t 不是标准 one-hot，row sum 范围: {row_sums.min()} ~ {row_sums.max()}")

    y_true = np.argmax(GT, axis=1).astype(np.int64)

    return A, y_true, GT.shape[1]


def build_one_pt(mat_path, save_path, time_index=7):
    A, y_true, num_classes = extract_time_slice(mat_path, time_index=time_index)

    n = A.shape[0]

    data_dict = {
        "A": torch.tensor(A[None, :, :], dtype=torch.float32),  # [1, N, N]
        "X_categorical": torch.empty((n, 0), dtype=torch.long),
        "X_poisson": torch.empty((n, 0), dtype=torch.float32),
        "X_gaussian": torch.empty((n, 0), dtype=torch.float32),
        "time": torch.tensor([time_index], dtype=torch.long),
        "y_true": torch.tensor(y_true, dtype=torch.long),
        "meta": {
            "dataset": "LFR",
            "source_mat": os.path.basename(mat_path),
            "time_index": int(time_index),
            "num_nodes": int(n),
            "num_classes": int(num_classes),
        }
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_dict, save_path)

    print(f"Saved: {save_path}")
    print(f"  A shape: {tuple(data_dict['A'].shape)}")
    print(f"  y_true shape: {tuple(data_dict['y_true'].shape)}")
    print(f"  num_classes: {num_classes}")
    print(f"  class counts (first 10): {torch.bincount(data_dict['y_true'])[:10].tolist()}")
    print(f"  num_edges (undirected): {int(A.sum() // 2)}")
    print(f"  avg_degree: {float(A.sum() / n):.4f}")
    print("-" * 60)


def main():
    # 你原始 .mat 文件所在目录
    in_dir = r"E:\Article\2025HIPC-improved\DiMAE\Datasets\olddata\LFR-data-new\1-data"

    # 输出 raw pt 文件目录
    out_dir = r"E:\Article\2026CausalDetection\master\data\input\LFR\raw_time7"

    file_ids = [101, 102, 103, 104, 105]
    time_index = 7

    for fid in file_ids:
        mat_path = os.path.join(in_dir, f"LFR-data{fid}.mat")
        save_path = os.path.join(out_dir, f"LFR_time{time_index}_raw_{fid}.pt")
        build_one_pt(mat_path, save_path, time_index=time_index)

    print("All done.")


if __name__ == "__main__":
    main()