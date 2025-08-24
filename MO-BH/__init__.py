# __init__.py

import numpy as np
import scipy.io as sio
import config as cfg
from precoder import HybridPrecoder_MU_MO
from tqdm import tqdm
from channel_estimation import OMPChannelEstimator
from quantization import quantize_channel_params
from wmmse import wmmse_precoder  # 导入新的WMMSE模块


def load_and_slice_channel_from_mat(filepath, num_samples, user_indices):
    """
    从.mat文件加载数据，处理4D MISO数据并扩展为5D，然后切片。
    """
    print(f"Loading channel data from {filepath}...")
    data = sio.loadmat(filepath)
    H_all_samples_raw = data['H_UPA']
    print(f"Original data shape from .mat file: {H_all_samples_raw.shape}")

    # 检查维度，如果是4D (samples, users, subcarriers, Nt)，则扩展为5D
    if H_all_samples_raw.ndim == 4:
        # 在第3个轴(axis=3)插入一个维度给 Nr=1
        H_all_samples_5d = np.expand_dims(H_all_samples_raw, axis=3)
        print(f"Reshaped to 5D shape (for Nr=1): {H_all_samples_5d.shape}")
    elif H_all_samples_raw.ndim == 5:
        H_all_samples_5d = H_all_samples_raw
    else:
        raise ValueError(f"Loaded data has an unexpected shape: {H_all_samples_raw.shape}")

    # 现在 H_all_samples_5d 是 (samples, users, subcarriers, Nr, Nt)
    # 之前对调维度的 transpose 不再需要，因为我们已经手动确保了正确的顺序
    # 如果你的 H_UPA.mat 是 (samples, users, Nr, Nt, subcarriers)，则需要 transpose
    # 根据你的 channel_OFDM.ipynb, 保存的维度是 (samples, users, subcarriers, Nt)，所以我们扩展后是 (samples, users, subcarriers, 1, Nt)

    H_sliced = H_all_samples_5d[:num_samples, user_indices, :, :, :]
    print(f"Loaded and sliced data shape: {H_sliced.shape}")
    return H_sliced


def calculate_sum_rate_mu_miso(H_list_users, F_RF, F_BB_list, noise_power):
    """为MU-MISO系统计算和速率"""
    total_rate = 0
    for n in range(cfg.Nc):
        F_n = F_RF @ F_BB_list[n]
        for k in range(cfg.K):
            # h_kn 是 (1, M) 的行向量
            h_kn = H_list_users[k][n, 0, :]

            signal_power = np.abs(h_kn @ F_n[:, k]) ** 2

            interference_power = 0
            for j in range(cfg.K):
                if j != k:
                    interference_power += np.abs(h_kn @ F_n[:, j]) ** 2

            sinr = signal_power / (interference_power + noise_power)
            total_rate += np.log2(1 + sinr)

    return total_rate / cfg.Nc


if __name__ == '__main__':
    user_indices_to_use = list(range(cfg.K))
    mat_filepath = '../data/H_UPA.mat'  # 请确保路径正确
    num_samples_to_test = 50

    H_dataset = load_and_slice_channel_from_mat(mat_filepath, num_samples_to_test, user_indices_to_use)

    precoder_algo = HybridPrecoder_MU_MO()
    channel_estimator = OMPChannelEstimator()

    results_perfect_csi = []
    results_estimated_csi = []
    results_limited_feedback = []

    noise_power = cfg.Pt / (cfg.Nc * cfg.SNR_linear)

    print(f"\nProcessing {num_samples_to_test} samples for 3 scenarios at SNR={cfg.SNR_dB}dB...")
    for i in tqdm(range(num_samples_to_test), desc="Simulating Samples"):
        H_list_users_true = [H_dataset[i, k, :, :, :] for k in range(cfg.K)]

        # --- 场景 1: Perfect CSI ---
        F_opt_list_wmmse = wmmse_precoder(H_list_users_true, noise_power)
        F_RF_p, F_BB_list_p = precoder_algo.compute(F_opt_list_wmmse)
        rate_p = calculate_sum_rate_mu_miso(H_list_users_true, F_RF_p, F_BB_list_p, noise_power)
        results_perfect_csi.append(rate_p)

        # --- 场景 2: Estimated CSI with Perfect Feedback ---
        estimated_params = channel_estimator.estimate_channel_parameters(H_list_users_true,
                                                                         noise_power)
        H_list_users_est = channel_estimator.reconstruct_channel_from_params(estimated_params)
        F_opt_list_e = wmmse_precoder(H_list_users_est, noise_power)
        F_RF_e, F_BB_list_e = precoder_algo.compute(F_opt_list_e)
        rate_e = calculate_sum_rate_mu_miso(H_list_users_true, F_RF_e, F_BB_list_e, noise_power)
        results_estimated_csi.append(rate_e)

        # --- 场景 3: Estimated CSI with Limited Feedback ---
        quantized_params = quantize_channel_params(estimated_params)
        H_list_users_recon = channel_estimator.reconstruct_channel_from_params(quantized_params)
        F_opt_list_q = wmmse_precoder(H_list_users_recon, noise_power)
        F_RF_q, F_BB_list_q = precoder_algo.compute(F_opt_list_q)
        rate_q = calculate_sum_rate_mu_miso(H_list_users_true, F_RF_q, F_BB_list_q, noise_power)
        results_limited_feedback.append(rate_q)

    avg_rate_perfect = np.mean(results_perfect_csi)
    avg_rate_estimated = np.mean(results_estimated_csi)
    avg_rate_limited = np.mean(results_limited_feedback)

    print("\n--- Final Simulation Results ---")
    print(f"Number of samples processed: {num_samples_to_test}")
    print(f"SNR: {cfg.SNR_dB} dB")
    print("-" * 30)
    print(f"Scenario 1 (Perfect CSI):    Average Sum Rate = {avg_rate_perfect:.4f} bps/Hz")
    print(f"Scenario 2 (Estimated CSI):  Average Sum Rate = {avg_rate_estimated:.4f} bps/Hz")
    print(f"Scenario 3 (Limited Fdbk):   Average Sum Rate = {avg_rate_limited:.4f} bps/Hz")