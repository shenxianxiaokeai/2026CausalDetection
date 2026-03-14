import os
import torch

# =========================
# 超参数
# =========================
EPS = 1e-3                 # propensity 下界，别太小
CLIP_Q = 0.95              # 权重裁剪分位数，先用 0.95 更稳
MAX_WEIGHT = 10.0          # 最大硬截断
SCALE_MODE = "mean"        # 可选: "mean" / "fro" / "none"


def estimate_propensity_from_graph(A_obs: torch.Tensor):
    """
    当输入文件中没有 E_true 时，
    使用 configuration-model 风格的近似 propensity 作为退化方案。
    """
    d_out = A_obs.sum(dim=1) + 1e-8
    d_in = A_obs.sum(dim=0) + 1e-8
    m = A_obs.sum() + 1e-8

    P = torch.outer(d_out, d_in) / m

    # 防止极端小概率
    P = torch.clamp(P, min=EPS, max=1.0 - EPS)

    eye = torch.eye(A_obs.size(0), dtype=torch.bool)
    P[eye] = 0.0
    return P


def normalize_ipw_to_obs(A_obs: torch.Tensor, A_ipw: torch.Tensor, mode: str = "mean"):
    """
    让 A_ipw 和 A_obs 尽量处在同一量级上。
    """
    if mode == "none":
        return A_ipw, 1.0

    if mode == "mean":
        obs_mean = A_obs.mean()
        ipw_mean = A_ipw.mean()
        scale = obs_mean / (ipw_mean + 1e-12)
        return A_ipw * scale, float(scale.item())

    if mode == "fro":
        obs_norm = torch.norm(A_obs, p="fro")
        ipw_norm = torch.norm(A_ipw, p="fro")
        scale = obs_norm / (ipw_norm + 1e-12)
        return A_ipw * scale, float(scale.item())

    raise ValueError(f"Unsupported SCALE_MODE: {mode}")


def main():
    # =========================
    # 路径设置
    # =========================
    in_path = os.path.join("input", "email-Eu-core", "raw", "EmailEU_raw.pt")
    out_path = os.path.join("input", "email-Eu-core", "corrected", "EmailEU_ipw.pt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # =========================
    # 加载数据
    # =========================
    data = torch.load(in_path, map_location="cpu")
    if "A" not in data:
        raise KeyError(f"在文件 {in_path} 中未找到键 'A'")

    A = data["A"].float()
    if A.dim() != 3 or A.size(0) < 1:
        raise ValueError(f"期望输入 A.shape = (L, N, N)，实际得到 {tuple(A.shape)}")

    # 只取第0层作为观测层
    A_obs = A[0].clone()

    # 再保险：对称化 + 去对角
    A_obs = 0.5 * (A_obs + A_obs.T)
    eye = torch.eye(A_obs.size(0), dtype=torch.bool)
    A_obs[eye] = 0.0

    # =========================
    # 选择 propensity 来源
    # =========================
    if "E_true" in data:
        P = data["E_true"].float()
        ipw_source = "E_true"
    else:
        P = estimate_propensity_from_graph(A_obs)
        ipw_source = "estimated"

    if P.shape != A_obs.shape:
        raise ValueError(
            f"P 与 A_obs 形状不一致: P={tuple(P.shape)}, A_obs={tuple(A_obs.shape)}"
        )

    # =========================
    # 构造原始 IPW 校正层
    # =========================
    denom = torch.clamp(P, min=EPS)
    obs_mask = A_obs > 0

    A_ipw = torch.zeros_like(A_obs)
    A_ipw[obs_mask] = A_obs[obs_mask] / denom[obs_mask]

    # 去对角
    A_ipw[eye] = 0.0

    # =========================
    # 第一步：分位裁剪
    # =========================
    if obs_mask.any():
        q_cap = torch.quantile(A_ipw[obs_mask], CLIP_Q)
        cap = min(float(q_cap.item()), MAX_WEIGHT)
        A_ipw = torch.clamp(A_ipw, max=cap)
    else:
        cap = 0.0

    # =========================
    # 第二步：尺度对齐到 A_obs
    # =========================
    A_ipw, scale = normalize_ipw_to_obs(A_obs, A_ipw, mode=SCALE_MODE)

    # =========================
    # 第三步：再做一次对称化
    # =========================
    A_ipw = 0.5 * (A_ipw + A_ipw.T)
    A_ipw[eye] = 0.0

    # =========================
    # 组织成两层输入：[观测层, 校正层]
    # =========================
    A_views = torch.stack([A_obs, A_ipw], dim=0)

    out_data = dict(data)
    out_data["A"] = A_views
    out_data["A_obs"] = A_obs
    out_data["A_ipw"] = A_ipw
    out_data["view_names"] = ["obs", "corrected"]
    out_data["ipw_source"] = ipw_source
    out_data["ipw_clip_q"] = CLIP_Q
    out_data["ipw_max_weight"] = MAX_WEIGHT
    out_data["ipw_scale_mode"] = SCALE_MODE
    out_data["ipw_scale_factor"] = scale

    torch.save(out_data, out_path)

    # =========================
    # 打印总结信息
    # =========================
    obs_mean = A_obs.mean().item()
    obs_max = A_obs.max().item()
    ipw_mean = A_ipw.mean().item()
    ipw_max = A_ipw.max().item()

    nonzero_ipw_mean = A_ipw[obs_mask].mean().item() if obs_mask.any() else 0.0

    print("========== IPW Layer Construction ==========")
    print(f"输入文件:  {in_path}")
    print(f"输出文件:  {out_path}")
    print(f"propensity来源: {ipw_source}")
    print(f"A_obs shape = {tuple(A_obs.shape)}")
    print(f"A_views shape = {tuple(A_views.shape)}")
    print(f"裁剪分位数 CLIP_Q = {CLIP_Q}")
    print(f"硬截断上限 MAX_WEIGHT = {MAX_WEIGHT}")
    print(f"尺度对齐方式 SCALE_MODE = {SCALE_MODE}")
    print(f"尺度因子 scale = {scale:.6f}")
    print("------ Layer stats ------")
    print(f"A_obs : mean = {obs_mean:.6f}, max = {obs_max:.6f}")
    print(f"A_ipw : mean = {ipw_mean:.6f}, max = {ipw_max:.6f}")
    print(f"IPW非零均值 = {nonzero_ipw_mean:.6f}")
    print(f"裁剪阈值 cap = {cap:.6f}")
    print("视图说明: A[0] = obs, A[1] = corrected")
    print("===========================================")
    print(f"[已保存] {out_path}")


if __name__ == "__main__":
    main()