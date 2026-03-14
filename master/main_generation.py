import os
import time
import yaml
import numpy as np
import torch
from argparse import ArgumentParser
from src.model import PIHAM, assign_priors


def main():
    # =========================
    # 1) 解析命令行参数
    # =========================
    p = ArgumentParser()
    p.add_argument("-f", "--in_folder", type=str, default="data/input/", help="输入数据所在目录")
    p.add_argument("-d", "--data_file", type=str, default="synthetic_data.pt", help="输入数据文件名(.pt)")
    p.add_argument("-K", "--K", type=int, default=4, help="社团数 K")
    args = p.parse_args()

    # =========================
    # 2) 输出目录准备
    # =========================
    output_folder = "data/output/"
    os.makedirs(output_folder, exist_ok=True)

    # =========================
    # 3) 读取推断配置（超参数、device等）
    # =========================
    with open("src/setting_inference.yaml", "r", encoding="utf-8") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    # device：建议转成 torch.device，并支持自动回退
    device_str = configuration.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warn] setting_inference.yaml 指定 cuda，但当前环境不可用，自动回退到 cpu")
        device_str = "cpu"
    device = torch.device(device_str)

    # 保存本次配置，便于复现实验
    with open(os.path.join(output_folder, "setting_inference.yaml"), "w", encoding="utf-8") as file:
        _ = yaml.dump(configuration, file, allow_unicode=True)

    # =========================
    # 4) 读取数据（A 必须有；X_* 可选）
    # =========================
    data_path = os.path.join(args.in_folder, args.data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件：{data_path}")

    data = torch.load(data_path)

    # ---- 必须：邻接张量 A，形状 (L, N, N) ----
    A = data["A"].to(device)

    # ---- 可选：协变量（允许“只有邻接矩阵”的数据集）----
    # 如果没有协变量，就用 (N, 0) 的空矩阵占位，避免后续报错
    num_nodes = A.size(1)
    X_categorical = data.get("X_categorical", torch.empty((num_nodes, 0))).to(device)
    X_poisson     = data.get("X_poisson",     torch.empty((num_nodes, 0))).to(device)
    X_gaussian    = data.get("X_gaussian",    torch.empty((num_nodes, 0))).to(device)

    # 这里把关键维度打印出来，便于你确认数据正确
    L = A.size(0)
    K = args.K
    Z_categorical = X_categorical.size(1)
    P_poisson     = X_poisson.size(1)
    P_gaussian    = X_gaussian.size(1)

    print(f"[Info] Loaded A shape = {tuple(A.shape)} (L,N,N)  L={L}, N={num_nodes}")
    print(f"[Info] Covariates dims: Z_cat={Z_categorical}, P_pois={P_poisson}, P_gauss={P_gaussian}")

    # =========================
    # 5) 运行模型：先验 -> 初始化 -> 自动识别层类型 -> 拟合
    # =========================
    tic = time.time()

    # 5.1) 根据 K/N/L/协变量维度构建先验
    priors = assign_priors(
        K, num_nodes, L, Z_categorical, P_poisson, P_gaussian, configuration
    )

    # 5.2) 初始化模型
    model = PIHAM(
        priors[0],  priors[6],   # U_mu, U_std
        priors[1],  priors[7],   # V_mu, V_std
        priors[2],  priors[8],   # W_mu, W_std
        priors[3],  priors[9],   # Hcat_mu, Hcat_std
        priors[4],  priors[10],  # Hpois_mu, Hpois_std
        priors[5],  priors[11],  # Hgauss_mu, Hgauss_std
        K, num_nodes, L,
        Z_categorical, P_poisson, P_gaussian,
        configuration,
    )

    # 5.3) 自动识别每一层 A 应该走哪种似然
    # GN+IPW 两层时，期望识别为：第0层 bernoulli，第1层 gaussian
    model.set_layer_types(A)
    if hasattr(model, "layer_types"):
        print("[Info] layer_types =", model.layer_types)

    # 5.4) 拟合
    model.fit(
        A, X_categorical, X_poisson, X_gaussian,
        gamma=configuration["gamma"],
        tolerance=configuration["tolerance"],
        num_iter=configuration["num_iter"],
        likelihood_weight=configuration["lik_weight"],
        learning_rate=configuration["learning_rate"],
        verbose=configuration["verbose"],
        N_seeds=configuration["N_seeds"],
    )

    # =========================
    # 6) 不确定性估计：Hessian -> 协方差（可选，但你这份脚本默认会做）
    # =========================
    model.compute_Hessian(
        A, X_categorical, X_poisson, X_gaussian,
        likelihood_weight=configuration["lik_weight"],
    )
    model.is_neg_Hessian_pos_def()
    model.compute_Covariance(eps=1e-6)

    # =========================
    # 7) 保存结果
    # =========================
    model.save_results(
        folder_name=output_folder,
        file_name=f"_{args.data_file.replace('.pt', '')}_K{K}",
    )

    toc = time.time()
    print(f"\n ---- Time elapsed: {np.round(toc - tic, 4)} seconds ----")


if __name__ == "__main__":
    main()
