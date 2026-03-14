# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import torch


def read_edge_list(edge_path: str):
    """
    读取 email-Eu-core.txt
    每行格式: u v
    原图是有向图，这里为了社区检测转成无向图
    """
    edges = []
    nodes = set()

    with open(edge_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = line.split()
            u = int(u)
            v = int(v)
            if u == v:
                continue
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)

    return edges, sorted(nodes)


def read_labels(label_path: str):
    """
    读取 email-Eu-core-department-labels.txt
    每行格式: node_id label
    """
    node_to_label = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            node_id, label = line.split()
            node_to_label[int(node_id)] = int(label)
    return node_to_label


def build_adj_and_labels(edges, nodes, node_to_label):
    """
    把原始节点 ID 重映射到 [0, n-1]，并构造：
      - 无向邻接矩阵 A
      - 单标签 y_true
    """
    # 只保留既在图中又有标签的节点
    nodes = [u for u in nodes if u in node_to_label]
    nodes = sorted(nodes)

    node2idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    A = np.zeros((n, n), dtype=np.float32)

    for u, v in edges:
        if u not in node2idx or v not in node2idx:
            continue
        i = node2idx[u]
        j = node2idx[v]
        A[i, j] = 1.0
        A[j, i] = 1.0   # 转成无向图

    np.fill_diagonal(A, 0.0)

    raw_labels = np.array([node_to_label[u] for u in nodes], dtype=np.int64)

    # 重新编号标签到 [0, C-1]
    unique_labels = sorted(np.unique(raw_labels).tolist())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_true = np.array([label_map[x] for x in raw_labels], dtype=np.int64)

    meta = {
        "dataset": "email-Eu-core",
        "num_nodes": int(n),
        "num_edges": int(A.sum() // 2),
        "avg_degree": float(A.sum() / n),
        "num_classes": int(len(unique_labels)),
        "directed_original": True,
        "converted_to_undirected": True,
    }

    return A, y_true, meta, node2idx


def save_pt(A: np.ndarray, y_true: np.ndarray, meta: dict, save_path: str):
    n = A.shape[0]

    out = {
        "A": torch.tensor(A[None, :, :], dtype=torch.float32),   # [1, N, N]
        "X_categorical": torch.empty((n, 0), dtype=torch.long),
        "X_poisson": torch.empty((n, 0), dtype=torch.float32),
        "X_gaussian": torch.empty((n, 0), dtype=torch.float32),
        "time": torch.tensor([1], dtype=torch.long),
        "y_true": torch.tensor(y_true, dtype=torch.long),
        "meta": meta,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, save_path)

    print(f"Saved to: {save_path}")
    print("A shape:", tuple(out["A"].shape))
    print("y_true shape:", tuple(out["y_true"].shape))
    print("num_classes:", len(np.unique(y_true)))
    print("class counts:", np.bincount(y_true))
    print("meta:", meta)


def save_node_map(node2idx, save_path: str):
    map_path = Path(save_path).with_suffix(".node_map.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        f.write("original_id\tnew_index\n")
        for old_id, new_id in node2idx.items():
            f.write(f"{old_id}\t{new_id}\n")
    print(f"Node mapping saved to: {map_path}")


def main():
    # ========= 你改这里 =========
    edge_path = r"E:\Article\2026CausalDetection\master\data\input\email-Eu-core\email-Eu-core.txt"
    label_path = r"E:\Article\2026CausalDetection\master\data\input\email-Eu-core\email-Eu-core-department-labels.txt"
    save_path = r"/data/input/email-Eu-core/raw/EmailEU_raw.pt"
    # ===========================

    print("Reading edge list...")
    edges, nodes = read_edge_list(edge_path)
    print(f"Loaded graph edges: {len(edges)}")
    print(f"Loaded graph nodes: {len(nodes)}")

    print("Reading labels...")
    node_to_label = read_labels(label_path)
    print(f"Loaded labels: {len(node_to_label)}")

    print("Building adjacency and labels...")
    A, y_true, meta, node2idx = build_adj_and_labels(edges, nodes, node_to_label)

    print("Saving pt file...")
    save_pt(A, y_true, meta, save_path)

    print("Saving node mapping...")
    save_node_map(node2idx, save_path)

    print("Done.")


if __name__ == "__main__":
    main()