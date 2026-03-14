import torch

d = torch.load("input/LFR/raw_time7/LFR_time7_raw_101.pt", map_location="cpu")

# 1) 先看这个 pt 里到底存了哪些字段
print("keys:", list(d.keys()))

# 2) 你原来的 A 统计（保留）
A = d["A"]
print("A shape:", tuple(A.shape))
for l in range(A.shape[0]):
    x = A[l]
    print(f"layer {l}: min={x.min().item():.6g}, max={x.max().item():.6g}, mean={x.float().mean().item():.6g}")

# 3) 检查“真值标签/真值参数”是否存在（常见命名都扫一遍）
candidate_keys = [
    "z", "z_true", "labels", "label", "y", "y_true", "gt", "ground_truth", "community", "communities",
    "A_true", "A_obs", "A_full", "A_raw", "mask", "R", "R_obs",
    "E_true", "E", "propensity", "propensity_true"
]
print("\n[Check candidates]")
for k in candidate_keys:
    if k in d:
        v = d[k]
        if torch.is_tensor(v):
            print(f"- {k}: tensor, shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"- {k}: type={type(v)}")

# 4) 如果有“真值社团标签”，打印类别分布（你最关心的真值）
label_key = None
for k in ["z_true", "z", "labels", "label", "y_true", "y", "gt", "ground_truth"]:
    if k in d and torch.is_tensor(d[k]):
        if d[k].dim() == 1:  # (N,) 最像节点标签
            label_key = k
            break

if label_key is not None:
    z = d[label_key].cpu()
    z = z.long() if z.dtype != torch.long else z
    uniq = torch.unique(z)
    print(f"\n[Found node labels] key='{label_key}', N={z.numel()}, num_classes={uniq.numel()}")
    # 统计每类数量（假设标签从0开始，不从0开始也能看出分布）
    for c in uniq.tolist():
        cnt = int((z == c).sum().item())
        print(f"  class {c}: {cnt}")
else:
    print("\n[No obvious node labels] 没找到形如 (N,) 的标签向量（可能没有真值社团标签，或命名不在候选列表里）。")

# 5) 如果有 E_true / A_true / A_obs / mask 之类，给出统计（用于评估 MNAR）
def _print_stats(name, x):
    x = x.float()
    print(f"{name}: shape={tuple(x.shape)}, min={x.min().item():.6g}, max={x.max().item():.6g}, mean={x.mean().item():.6g}")

for k in ["E_true", "E", "mask", "R", "R_obs", "A_true", "A_obs"]:
    if k in d and torch.is_tensor(d[k]):
        print()
        _print_stats(k, d[k])
