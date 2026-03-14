# -*- coding: utf-8 -*-
"""
将 SNAP com-DBLP 处理成当前项目可直接使用的 .pt 文件

输入：
  - com-dblp.ungraph.txt
  - com-dblp.all.cmty.txt   或 com-dblp.top5000.cmty.txt

输出：
  - DBLP_raw_singlelabel.pt

说明：
1. SNAP 社区是重叠社区，而当前评估脚本要求单标签 y_true
2. 因此这里采用“只保留恰好属于一个社区的节点”的策略
3. 最终得到一个单标签诱导子图，可直接用于你现在的训练/评估
"""

import os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import networkx as nx


def read_ungraph_txt(edge_path: str):
    """
    读取 SNAP 无向图边文件
    格式通常是：
      # 注释
      u v
    """
    edges = []
    nodes = set()

    with open(edge_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = line.split()
            u = int(u)
            v = int(v)
            if u == v:
                continue
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)

    return edges, nodes


def read_cmty_txt(cmty_path: str):
    """
    读取 SNAP 社区文件
    每一行是一个社区，里面是节点 ID 列表
    """
    communities = []
    with open(cmty_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nodes = [int(x) for x in line.split()]
            if len(nodes) > 0:
                communities.append(nodes)
    return communities


def build_singlelabel_subgraph(edges, communities, min_comm_size=3):
    """
    从重叠社区中抽取“单标签节点”构成子图：
    - 只保留恰好属于一个社区的节点
    - 且该社区大小至少 min_comm_size
    """
    membership = defaultdict(list)

    for cid, comm in enumerate(communities):
        if len(comm) < min_comm_size:
            continue
        for u in comm:
            membership[u].append(cid)

    # 只保留恰好属于一个社区的节点
    single_nodes = {u for u, cids in membership.items() if len(cids) == 1}

    # 节点 -> 唯一社区
    node_label = {}
    for u in single_nodes:
        node_label[u] = membership[u][0]

    # 诱导边
    sub_edges = []
    for u, v in edges:
        if u in single_nodes and v in single_nodes:
            sub_edges.append((u, v))

    # 构图并取最大连通分量，避免碎图
    G = nx.Graph()
    G.add_edges_from(sub_edges)

    if G.number_of_nodes() == 0:
        raise ValueError("筛选后子图为空，请换社区文件或放宽筛选条件。")

    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()

    # 最终节点集合
    final_nodes = sorted(H.nodes())

    # 社区重新编号
    raw_labels = [node_label[u] for u in final_nodes]
    unique_labels = sorted(set(raw_labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_true = np.array([label_map[x] for x in raw_labels], dtype=np.int64)

    # 节点重新映射到 [0, n-1]
    node_id_map = {old: new for new, old in enumerate(final_nodes)}
    n = len(final_nodes)

    A = np.zeros((n, n), dtype=np.float32)
    for u, v in H.edges():
        i = node_id_map[u]
        j = node_id_map[v]
        A[i, j] = 1.0
        A[j, i] = 1.0

    np.fill_diagonal(A, 0.0)

    meta = {
        "original_num_nodes_in_graph": int(len(set(sum(([u, v] for u, v in edges), [])))),
        "num_nodes_after_singlelabel_filter": int(n),
        "num_edges_after_singlelabel_filter": int(A.sum() // 2),
        "num_classes_after_singlelabel_filter": int(len(unique_labels)),
        "kept_only_single_membership_nodes": True,
        "largest_connected_component_only": True,
    }

    return A, y_true, meta


def save_pt(A, y_true, meta, save_path):
    n = A.shape[0]
    out = {
        "A": torch.tensor(A[None, :, :], dtype=torch.float32),  # [1, N, N]
        "X_categorical": torch.empty((n, 0), dtype=torch.long),
        "X_poisson": torch.empty((n, 0), dtype=torch.float32),
        "X_gaussian": torch.empty((n, 0), dtype=torch.float32),
        "time": torch.tensor([1], dtype=torch.long),
        "y_true": torch.tensor(y_true, dtype=torch.long),
        "meta": meta,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, save_path)

    print(f"Saved to: {save_path}")
    print("A shape:", tuple(out["A"].shape))
    print("y_true shape:", tuple(out["y_true"].shape))
    print("num_classes:", len(np.unique(y_true)))
    print("class counts:", np.bincount(y_true))
    print("meta:", meta)


def main():
    # ========= 你改这里 =========
    edge_path = r"E:\Article\2026CausalDetection\master\data\input\SNAP\com-dblp.ungraph.txt\com-dblp.ungraph.txt"
    cmty_path = r"E:\Article\2026CausalDetection\master\data\input\SNAP\com-dblp.top5000.cmty.txt~\com-dblp.top5000.cmty.txt"
    save_path = r"E:\Article\2026CausalDetection\master\data\input\DBLP\DBLP_raw_singlelabel.pt"
    # ===========================

    print("Reading edges...")
    edges, nodes = read_ungraph_txt(edge_path)
    print(f"Graph loaded: {len(nodes)} nodes, {len(edges)} edges")

    print("Reading communities...")
    communities = read_cmty_txt(cmty_path)
    print(f"Community file loaded: {len(communities)} communities")

    print("Building single-label induced subgraph...")
    A, y_true, meta = build_singlelabel_subgraph(edges, communities, min_comm_size=3)

    save_pt(A, y_true, meta, save_path)


if __name__ == "__main__":
    main()