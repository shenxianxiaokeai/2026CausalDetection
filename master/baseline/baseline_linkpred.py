import os
import argparse
import numpy as np
import torch

def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUC via rank (Mann–Whitney U). No sklearn needed.
    labels: 0/1
    """
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)

    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    # ranks with tie handling: average ranks for ties
    all_scores = np.concatenate([pos, neg], axis=0)
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1, dtype=np.float64)

    # tie correction
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg_rank
        i = j

    n_pos = len(pos)
    n_neg = len(neg)
    sum_ranks_pos = ranks[:n_pos].sum()  # because pos are first in concatenation
    # Actually ranks computed on concatenation [pos, neg]
    # So pos indices are [0..n_pos-1]
    sum_ranks_pos = ranks[np.arange(n_pos)].sum()

    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auc = u / (n_pos * n_neg + 1e-12)
    return float(auc)

def get_key(data: dict, candidates):
    for k in candidates:
        if k in data:
            return k
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_folder", type=str, default="../data/input")
    ap.add_argument("--data_file", type=str, default="GN_time4_mnar_raw.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    path = os.path.join(args.in_folder, args.data_file)
    data = torch.load(path, map_location="cpu")

    # 尽量兼容你 pt 里可能的命名
    k_true = get_key(data, ["A_true", "Atrue", "adj_true", "A_gt", "Agt"])
    k_obs  = get_key(data, ["A_obs", "Aobs", "adj_obs", "A_raw", "Araw", "A"])
    if k_true is None:
        raise KeyError(f"cannot find A_true in keys={list(data.keys())}")
    if k_obs is None:
        raise KeyError(f"cannot find A_obs/A in keys={list(data.keys())}")

    A_true = data[k_true]
    A_obs = data[k_obs]

    # 处理形状：可能是 (1,N,N)
    if A_true.dim() == 3:
        A_true = A_true[0]
    if A_obs.dim() == 3:
        # 注意：如果 A_obs 用 "A" 存的是 (L,N,N)，这里默认 L=1
        A_obs = A_obs[0]

    A_true = (A_true > 0).to(torch.float32)
    A_obs = (A_obs > 0).to(torch.float32)

    N = A_true.size(0)
    eye = torch.eye(N, dtype=torch.bool)
    mask = ~eye

    # missing true edges
    miss_pos = torch.nonzero((A_true > 0.5) & (A_obs < 0.5) & mask, as_tuple=False)
    n_miss = miss_pos.size(0)

    # sampled non-edges from true graph
    rng = np.random.default_rng(args.seed)
    non_edges = torch.nonzero((A_true < 0.5) & mask, as_tuple=False)
    if n_miss > non_edges.size(0):
        raise ValueError("not enough true non-edges to sample")
    perm = rng.choice(non_edges.size(0), size=n_miss, replace=False)
    samp_neg = non_edges[perm]

    def eval_score(name, S: torch.Tensor):
        # all-pairs AUC for predicting A_true
        y_all = A_true[mask].numpy()
        s_all = S[mask].numpy()
        auc_all = auc_roc(s_all, y_all)

        # missing-vs-nonedges AUC
        y_bin = np.concatenate([np.ones(n_miss, dtype=np.int64), np.zeros(n_miss, dtype=np.int64)], axis=0)
        s_pos = S[miss_pos[:,0], miss_pos[:,1]].numpy()
        s_neg = S[samp_neg[:,0], samp_neg[:,1]].numpy()
        s_bin = np.concatenate([s_pos, s_neg], axis=0)
        auc_miss = auc_roc(s_bin, y_bin)

        print(f"{name:>10s} | AUC(all)={auc_all:.6f} | AUC(missing)={auc_miss:.6f} | score stats: mean={float(S[mask].mean()):.6f} std={float(S[mask].std()):.6f}")

    # ===== baselines =====
    A = A_obs.clone()
    A.fill_diagonal_(0)

    # baseline 0: use A_obs itself as score
    eval_score("A_obs", A)

    deg = A.sum(dim=1)  # (N,)
    # Preferential Attachment: deg_i * deg_j
    PA = deg[:, None] * deg[None, :]
    PA.fill_diagonal_(0)
    eval_score("PA", PA)

    # Common Neighbors: A @ A
    CN = A @ A
    CN.fill_diagonal_(0)
    eval_score("CN", CN)

    # Adamic-Adar: A @ diag(1/log(deg)) @ A
    deg_np = deg.numpy()
    w = np.zeros_like(deg_np, dtype=np.float32)
    for i, d in enumerate(deg_np):
        if d > 1:
            w[i] = 1.0 / np.log(d + args.eps)
        else:
            w[i] = 0.0
    W = torch.from_numpy(w)
    AA = (A * W[None, :]) @ A  # A @ diag(W) @ A
    AA.fill_diagonal_(0)
    eval_score("AA", AA)

    # Jaccard: CN / (deg_i + deg_j - CN)
    denom = deg[:, None] + deg[None, :] - CN + args.eps
    J = CN / denom
    J.fill_diagonal_(0)
    eval_score("Jaccard", J)

    print("\nInfo:")
    print(f"data={path}")
    print(f"N={N}, A_true mean={float(A_true[mask].mean()):.6f}, A_obs mean={float(A_obs[mask].mean()):.6f}")
    print(f"missing_true_edges={n_miss}")

if __name__ == "__main__":
    main()
