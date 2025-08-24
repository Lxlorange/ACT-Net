import numpy as np
import scipy.io as sio
import config as cfg
from precoder import HybridPrecoder_MU_MO, calculate_sum_rate
from tqdm import tqdm


def load_and_slice_channel_from_mat(filepath, num_samples, user_indices):
    """
    从.mat文件加载数据，并根据指定的索引选择用户。
    """
    print(f"Loading channel data from {filepath}...")
    data = sio.loadmat(filepath)
    H_all_samples = data['H_UPA']

    # 验证请求的用户数是否超过数据中的用户数
    if max(user_indices) >= H_all_samples.shape[1]:
        raise ValueError(
            f"User index {max(user_indices)} is out of bounds for data with {H_all_samples.shape[1]} users.")

    H_sliced = H_all_samples[:num_samples, user_indices, :, :]
    print(f"Loaded and sliced data shape: {H_sliced.shape}")
    return H_sliced


def compute_F_opt_svd(H_list):
    """
    计算理想预编码器 F_opt。使用SVD方法，因为它为MO算法提供了结构良好的酉矩阵目标。
    """
    F_opt_list = []
    for n in range(cfg.Nc):
        H_n = H_list[n]
        try:
            U, _, _ = np.linalg.svd(H_n, full_matrices=False)
        except np.linalg.LinAlgError:
            U, _ = np.linalg.qr(np.random.randn(H_n.shape[0], H_n.shape[1]))

        F_svd = U
        power_scale = np.sqrt(cfg.Pt / cfg.Nc) / np.linalg.norm(F_svd, 'fro')
        F_opt_list.append(F_svd * power_scale)
    return F_opt_list


if __name__ == '__main__':
    # --- 1. 参数设置 ---
    user_indices_to_use = [0, 1]
    # 确保config.py中的K与这里的用户数一致
    if cfg.K != len(user_indices_to_use):
        print(f"Warning: config.py has K={cfg.K}, but we are running a K={len(user_indices_to_use)} simulation.")
        cfg.K = len(user_indices_to_use)

    mat_filepath = '../data/H_UPA.mat'
    num_samples_to_test = 200  # 增加样本数量以获得更稳定的统计结果

    # --- 2. 加载并切片数据 ---
    H_dataset = load_and_slice_channel_from_mat(mat_filepath, num_samples_to_test, user_indices_to_use)

    # --- 3. 初始化算法和结果记录 ---
    precoder_algo = HybridPrecoder_MU_MO()
    sum_rate_results = []
    noise_power = cfg.Pt / (cfg.Nc * cfg.SNR_linear)

    # --- 4. 循环处理每个信道样本 ---
    print(f"\nProcessing {num_samples_to_test} samples with MO-AltMin (SVD target)...")
    for i in tqdm(range(num_samples_to_test), desc="Simulating Samples"):
        H_sample = H_dataset[i]
        H_list = [H_sample[:, n, :].T for n in range(cfg.Nc)]

        F_opt_list = compute_F_opt_svd(H_list)

        # 对每个样本独立运行算法
        F_RF, F_BB_list = precoder_algo.compute(F_opt_list)

        rate = calculate_sum_rate(H_list, F_RF, F_BB_list, noise_power)
        sum_rate_results.append(rate)

    # --- 5. 计算并打印最终平均结果 ---
    average_sum_rate = np.mean(sum_rate_results)

    print("\n--- Final Simulation Results ---")
    print(f"Number of samples processed: {num_samples_to_test}")
    print(f"SNR: {cfg.SNR_dB} dB")
    print(f"Average Sum Rate: {average_sum_rate:.4f} bps/Hz")