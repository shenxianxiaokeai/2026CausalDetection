import os
import torch


def solve_b0_for_target_mean(alpha: float,
                             x: torch.Tensor,
                             target: float,
                             mask: torch.Tensor,
                             iters: int = 60) -> float:
    """
    求解方程中的b0值：
        E_ij = sigmoid(b0 + alpha*x_i + alpha*x_j)
    使得有效元素上的E_ij均值接近目标值target。
    """
    lo, hi = -30.0, 30.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        E = torch.sigmoid(mid + alpha * x[:, None] + alpha * x[None, :])
        m = E[mask].mean().item()
        if m < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_prob_matrix(mode: str,
                      A_true: torch.Tensor,
                      mask: torch.Tensor,
                      target_obs_rate: float,
                      alpha: float,
                      e_min: float,
                      e_max: float,
                      seed: int):
    """
    根据缺失机制类型构造观测概率矩阵 E_true

    mode:
        - "MNAR": 依赖潜在结构（这里用真实图度特征）
        - "MAR" : 依赖外生节点协变量
        - "MCAR": 完全随机缺失
    """
    N = A_true.size(0)
    eye = ~mask  # 对角线位置

    if mode == "MNAR":
        # 基于真实图度构造潜在特征
        deg = A_true.sum(dim=1)
        x = torch.log1p(deg)
        b0 = solve_b0_for_target_mean(alpha, x, target_obs_rate, mask)
        E_true = torch.sigmoid(b0 + alpha * x[:, None] + alpha * x[None, :])
        aux_info = {"node_score": x}

    elif mode == "MAR":
        # 构造一个外生节点协变量，不依赖边是否存在
        g = torch.Generator().manual_seed(seed + 12345)
        x = torch.randn(N, generator=g)
        x = (x - x.mean()) / (x.std() + 1e-8)

        b0 = solve_b0_for_target_mean(alpha, x, target_obs_rate, mask)
        E_true = torch.sigmoid(b0 + alpha * x[:, None] + alpha * x[None, :])
        aux_info = {"node_score": x}

    elif mode == "MCAR":
        # 完全随机：所有非对角线位置同一观测概率
        E_true = torch.full_like(A_true, fill_value=target_obs_rate)
        aux_info = {}

    else:
        raise ValueError(f"不支持的缺失机制类型: {mode}")

    E_true = torch.clamp(E_true, e_min, e_max)
    E_true[eye] = 0.0
    return E_true, aux_info


def sample_observed_graph(A_true: torch.Tensor,
                          E_true: torch.Tensor,
                          seed: int):
    """
    根据观测概率矩阵 E_true 采样得到无向观测图 A_obs
    """
    N = A_true.size(0)
    eye = torch.eye(N, dtype=torch.bool)

    g = torch.Generator().manual_seed(seed)
    U = torch.rand((N, N), generator=g)

    M = (U < E_true).float()
    M = torch.triu(M, diagonal=1)
    M = M + M.t()

    A_obs = (A_true * M).clamp(0.0, 1.0)
    A_obs[eye] = 0.0
    return A_obs


def save_missing_data(mode: str,
                      A_true: torch.Tensor,
                      A_obs: torch.Tensor,
                      E_true: torch.Tensor,
                      data: dict,
                      out_path: str,
                      aux_info: dict):
    """
    保存缺失观测图数据
    """
    out_data = {
        "A": A_obs.unsqueeze(0),   # shape: (1, N, N)
        "A_true": A_true,
        "A_obs": A_obs,
        "E_true": E_true,
        "missing_type": mode,
        **aux_info,
        **{k: v for k, v in data.items() if k != "A"},
    }

    torch.save(out_data, out_path)
    print(f"[已保存 {mode}] {out_path}")


def print_summary(mode: str,
                  in_path: str,
                  out_path: str,
                  N: int,
                  seed: int,
                  target_obs_rate: float,
                  alpha: float,
                  E_true: torch.Tensor,
                  A_true: torch.Tensor,
                  A_obs: torch.Tensor,
                  mask: torch.Tensor):
    """
    打印当前缺失机制的统计信息
    """
    retained_edges = (A_obs.sum() / (A_true.sum() + 1e-8)).item()

    print(f"\n========== {mode} 缺失数据生成结果 ==========")
    print(f"输入文件:  {in_path}")
    print(f"输出文件: {out_path}")
    print(f"节点数量 N = {N}, 随机种子 seed = {seed}")
    print(f"目标观测率 target_obs_rate = {target_obs_rate}, 强度 alpha = {alpha}")
    print(
        f"观测概率矩阵E_true的统计（非对角线）= "
        f"均值({E_true[mask].mean().item():.4f}, "
        f"最小值{E_true[mask].min().item():.4f}, "
        f"最大值{E_true[mask].max().item():.4f})"
    )
    print(f"原始图A_true的边数 = {int(A_true.sum().item())}")
    print(f"观测图A_obs的边数 = {int(A_obs.sum().item())}")
    print(f"保留的边比例 = {retained_edges:.4f}")
    print("===========================================")


def main():
    # =========================
    # 路径设置
    # =========================
    in_path = os.path.join("input", "GN", "raw", "GN_time4_raw.pt")
    out_dir = os.path.join("input", "GN", "raw")
    os.makedirs(out_dir, exist_ok=True)

    out_paths = {
        "MNAR": os.path.join(out_dir, "GN_time4_mnar_raw.pt"),
        "MAR":  os.path.join(out_dir, "GN_time4_mar_raw.pt"),
        "MCAR": os.path.join(out_dir, "GN_time4_mcar_raw.pt"),
    }

    # =========================
    # 超参数设置
    # =========================
    seed = 0
    target_obs_rate = 0.6
    alpha = 0.8
    e_min, e_max = 0.05, 0.95

    # =========================
    # 加载数据
    # =========================
    data = torch.load(in_path, map_location="cpu")
    if "A" not in data:
        raise KeyError(f"在文件 {in_path} 中未找到键'A'")

    A = data["A"]
    if A.dim() != 3 or A.size(0) != 1:
        raise ValueError(f"期望A的形状为 (1, N, N)，实际得到 {tuple(A.shape)}")

    A_true = A[0].float()
    N = A_true.size(0)

    # 去除自环
    eye = torch.eye(N, dtype=torch.bool)
    mask = ~eye
    A_true = A_true.clone()
    A_true[eye] = 0.0

    # =========================
    # 分别生成 MNAR / MAR / MCAR
    # =========================
    for mode in ["MNAR", "MAR", "MCAR"]:
        E_true, aux_info = build_prob_matrix(
            mode=mode,
            A_true=A_true,
            mask=mask,
            target_obs_rate=target_obs_rate,
            alpha=alpha,
            e_min=e_min,
            e_max=e_max,
            seed=seed,
        )

        A_obs = sample_observed_graph(
            A_true=A_true,
            E_true=E_true,
            seed=seed,
        )

        print_summary(
            mode=mode,
            in_path=in_path,
            out_path=out_paths[mode],
            N=N,
            seed=seed,
            target_obs_rate=target_obs_rate,
            alpha=alpha,
            E_true=E_true,
            A_true=A_true,
            A_obs=A_obs,
            mask=mask,
        )

        save_missing_data(
            mode=mode,
            A_true=A_true,
            A_obs=A_obs,
            E_true=E_true,
            data=data,
            out_path=out_paths[mode],
            aux_info=aux_info,
        )


if __name__ == "__main__":
    main()