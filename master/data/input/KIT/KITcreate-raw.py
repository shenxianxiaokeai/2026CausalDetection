import json
import torch
import numpy as np


def load_time_entry(json_path, target_time):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if item.get("time") == target_time:
            return item

    raise ValueError(f"在 {json_path} 中没有找到 time={target_time}")


def build_time_slice_pt(
    graph_json_path="kit2.json",
    truth_json_path="kit2_truth.json",
    target_time=8,
    save_path="KIT_time8_raw.pt",
):
    # 1) 图
    graph_item = load_time_entry(graph_json_path, target_time)
    A = np.array(graph_item["matrix"], dtype=np.float32)

    # 2) 真值 one-hot
    truth_item = load_time_entry(truth_json_path, target_time)
    Y = np.array(truth_item["matrix"], dtype=np.float32)

    if Y.ndim != 2:
        raise ValueError(f"truth matrix 应为二维，当前 shape={Y.shape}")

    # 3) one-hot -> hard labels
    y_true = np.argmax(Y, axis=1).astype(np.int64)

    # 4) 一些检查
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A 不是方阵，当前 shape={A.shape}")
    if len(y_true) != n:
        raise ValueError(f"标签数 {len(y_true)} 与节点数 {n} 不一致")

    num_classes = int(len(np.unique(y_true)))

    data_dict = {
        "A": torch.tensor(A[None, :, :], dtype=torch.float32),   # [1, N, N]
        "X_categorical": torch.empty((n, 0), dtype=torch.long),
        "X_poisson": torch.empty((n, 0), dtype=torch.float32),
        "X_gaussian": torch.empty((n, 0), dtype=torch.float32),
        "time": torch.tensor([target_time], dtype=torch.long),
        "y_true": torch.tensor(y_true, dtype=torch.long),
    }

    torch.save(data_dict, save_path)

    print(f"Saved to: {save_path}")
    print("A shape:", data_dict["A"].shape)
    print("y_true shape:", data_dict["y_true"].shape)
    print("num_classes:", num_classes)
    print("class counts:", torch.bincount(data_dict["y_true"]))


if __name__ == "__main__":
    build_time_slice_pt(
        graph_json_path="kit2.json",
        truth_json_path="kit2_truth.json",
        target_time=7,
        save_path="raw/KIT_time7_raw.pt",
    )