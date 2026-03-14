import json
import torch
import numpy as np


def load_time_entry(json_path, target_time):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if item.get("time") == target_time:
            return item

    raise ValueError(f"在 {json_path} 中没有找到 time={target_time} 的数据")


def build_gn_time4_raw_pt(
    graph_json_path="gndata1.json",
    truth_json_path="gndata1_truth.json",
    target_time=4,
    save_path="GN_time4_raw.pt",
):
    # 1) 读取 time=4 的图邻接矩阵
    graph_item = load_time_entry(graph_json_path, target_time)
    A = np.array(graph_item["matrix"], dtype=np.float32)

    # 2) 读取 time=4 的真实标签 one-hot 矩阵
    truth_item = load_time_entry(truth_json_path, target_time)
    Y = np.array(truth_item["matrix"], dtype=np.float32)

    # 3) one-hot -> hard labels
    y_true = np.argmax(Y, axis=1).astype(np.int64)

    # 4) 按你当前项目格式组织
    data_dict = {
        "A": torch.tensor(A[None, :, :], dtype=torch.float32),   # shape [1, N, N]
        "X_categorical": torch.empty((A.shape[0], 0), dtype=torch.long),
        "X_poisson": torch.empty((A.shape[0], 0), dtype=torch.float32),
        "X_gaussian": torch.empty((A.shape[0], 0), dtype=torch.float32),
        "time": torch.tensor([target_time], dtype=torch.long),
        "y_true": torch.tensor(y_true, dtype=torch.long),
    }

    torch.save(data_dict, save_path)

    print(f"Saved to: {save_path}")
    print("A shape:", data_dict["A"].shape)
    print("y_true shape:", data_dict["y_true"].shape)
    print("Class counts:", torch.bincount(data_dict["y_true"]))


if __name__ == "__main__":
    build_gn_time4_raw_pt(
        graph_json_path="gndata1.json",
        truth_json_path="gndata1_truth.json",
        target_time=4,
        save_path="raw/GN_time4_raw.pt",
    )