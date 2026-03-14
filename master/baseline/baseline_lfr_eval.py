# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import torch
import networkx as nx

from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community.quality import modularity

warnings.filterwarnings("ignore")


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def labels_to_communities(labels):
    labels = np.asarray(labels)
    comms = []
    for c in np.unique(labels):
        comms.append(set(np.where(labels == c)[0]))
    return comms


def compute_modularity(A, labels):
    A = to_numpy(A)
    G = nx.from_numpy_array(A)
    comms = labels_to_communities(labels)
    return modularity(G, comms, weight="weight")


def eval_partition(y_true, y_pred, A):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    mod = compute_modularity(A, y_pred)
    return {
        "NMI": float(nmi),
        "ARI": float(ari),
        "Modularity": float(mod),
        "num_pred_classes": int(len(np.unique(y_pred))),
    }


def run_spectral(A, K, random_state=0):
    # A 要求是对称相似矩阵
    model = SpectralClustering(
        n_clusters=K,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=10,
    )
    y_pred = model.fit_predict(A)
    return y_pred


def run_louvain(A, resolution=1.0, seed=0):
    """
    需要 networkx >= 3.0，且内置 louvain_communities
    """
    G = nx.from_numpy_array(A)
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)

    y_pred = np.zeros(A.shape[0], dtype=np.int64)
    for cid, nodes in enumerate(comms):
        for v in nodes:
            y_pred[v] = cid
    return y_pred


def run_leiden(A, resolution=1.0, seed=0):
    """
    Leiden 依赖 python-igraph 和 leidenalg：
        pip install python-igraph leidenalg
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise ImportError(
            "运行 Leiden 需要安装: pip install python-igraph leidenalg"
        ) from e

    A = np.asarray(A)
    src, dst = np.where(np.triu(A, 1) > 0)
    weights = A[src, dst].tolist()

    g = ig.Graph(n=A.shape[0], edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    if len(weights) > 0:
        g.es["weight"] = weights

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if len(weights) > 0 else None,
        resolution_parameter=resolution,
        seed=seed,
    )

    y_pred = np.array(part.membership, dtype=np.int64)
    return y_pred


def main():
    pt_path = r"E:\Article\2026CausalDetection\master\data\input\LFR\raw_time7\LFR_time7_raw_101.pt"

    data = torch.load(pt_path, map_location="cpu")
    A = to_numpy(data["A"])[0]     # raw 只有一层 [1, N, N]
    y_true = to_numpy(data["y_true"]).astype(int)

    print("===== Data Summary =====")
    print("A shape:", A.shape)
    print("y_true shape:", y_true.shape)
    print("num_true_classes:", len(np.unique(y_true)))
    print("class counts:", np.bincount(y_true))
    print("num_edges:", int(A.sum() // 2))
    print("avg_degree:", float(A.sum() / A.shape[0]))
    print()

    K = len(np.unique(y_true))

    # ======================================================
    # 1) Spectral Clustering
    # ======================================================
    print("===== Spectral Clustering =====")
    y_pred_spec = run_spectral(A, K=K, random_state=0)
    spec_res = eval_partition(y_true, y_pred_spec, A)
    print(spec_res)
    print("pred bincount:", np.bincount(y_pred_spec))
    print()

    # ======================================================
    # 2) Louvain
    # ======================================================
    print("===== Louvain =====")
    y_pred_louvain = run_louvain(A, resolution=1.0, seed=0)
    louvain_res = eval_partition(y_true, y_pred_louvain, A)
    print(louvain_res)
    print("pred bincount:", np.bincount(y_pred_louvain))
    print()

    # ======================================================
    # 3) Leiden
    # ======================================================
    print("===== Leiden =====")
    try:
        y_pred_leiden = run_leiden(A, resolution=1.0, seed=0)
        leiden_res = eval_partition(y_true, y_pred_leiden, A)
        print(leiden_res)
        print("pred bincount:", np.bincount(y_pred_leiden))
    except Exception as e:
        print("Leiden 未运行成功：", str(e))

    print("\nDone.")


if __name__ == "__main__":
    main()