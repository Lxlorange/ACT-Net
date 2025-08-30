import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

# 导入您项目中的自定义函数
from mano import MO_AltMin
from codebook import generate_upa_codebook
from swomp import swomp_channel_estimation, reconstruct_channel
from util import *

# --- 仿真参数设置 ---
Nt = 64
Nr = 1
K = 2  # 用户数
Ns = K  # 每个用户一个数据流，总数据流数
NRF = K  # 射频链数量
Nc = 32  # 子载波数
L = 2  # 信道稀疏度 (路径数)
Q = 2 #导频长度

SNR_dB = np.arange(5, 16, 5)
SNR_lin = 10 ** (SNR_dB / 10)

# --- 数据加载 ---
data = loadmat("../data/H_UPA_4.mat")
H_dataset = data['H_UPA']

BATCH_SIZE = H_dataset.shape[0]
print(f"加载数据集 H_UPA, 维度: {H_dataset.shape}")
H_dataset = H_dataset[:, :K, :, :]  # (BATCH, K, Nc, Nt)

# --- 码本和导频生成 ---
Nt_params = (8, 8)
angle_samples = 64
At_codebook = generate_upa_codebook(Nt_params, angle_samples)
random_phases = 2 * np.pi * np.random.rand(Q, Nt)
X_pilot = (1 / np.sqrt(Nt)) * np.exp(1j * random_phases)
print(f"码本生成完毕，维度: {At_codebook.shape}")
print(f"导频生成完毕，维度: {X_pilot.shape}")

sumRatePerfect = np.zeros((len(SNR_dB), BATCH_SIZE))
sumRateEstimated = np.zeros((len(SNR_dB), BATCH_SIZE))

for si in range(BATCH_SIZE):
    print(f"正在处理样本: {si + 1}/{BATCH_SIZE}")
    H_sample_user_view = H_dataset[si, :, :, :]  # (K, Nc, Nt)
    # 将信道矩阵转换为 (K*Nr, Nc, Nt) 的形式，以便 get_Fopt_Wopt 处理
    H_sample_system_view = H_sample_user_view.reshape(K * Nr, Nc, Nt)

    # --- 路径1: 完美CSI下的波束赋形 ---
    Fopt_p, Wopt_p = get_Fopt_Wopt(H_sample_system_view, K, Nr, Ns)
    FRF_p, FBB_p = MO_AltMin(Fopt_p, NRF)

    # 在计算速率前，对整个批次的预编码器进行功率归一化
    for k in range(Nc):
        norm_f = np.linalg.norm(FRF_p @ FBB_p[:, :, k], 'fro')
        if norm_f > 1e-9:
            # 缩放FBB以满足功率约束 ||FRF*FBB||_F^2 = Ns
            FBB_p[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB_p[:, :, k]

    # --- 遍历所有SNR点 ---
    for s_idx, snr_val in enumerate(SNR_lin):

        # --- 路径1的速率计算 ---
        # 使用修正后的函数，传入Wopt_p
        rate_p = calculate_sum_rate(H_sample_user_view, FRF_p, FBB_p, Wopt_p, snr_val, K, Ns, Nc)
        sumRatePerfect[s_idx, si] = rate_p

        # --- 信道估计过程 ---
        # 1. 生成无噪声的接收导频
        # H_sample_system_view: (K*Nr, Nc, Nt), X_pilot.T: (Nt, Q) -> Y_noiseless: (K*Nr, Nc, Q)
        Y_noiseless = np.tensordot(H_sample_system_view, X_pilot.T, axes=([2], [0]))

        # 2. 添加噪声
        # snr_val 定义为每根接收天线的接收信号功率与噪声功率之比
        # 这里我们假设导频信号的平均功率为1
        noise_variance = 1 / snr_val
        noise = np.sqrt(noise_variance / 2) * (
                np.random.randn(*Y_noiseless.shape) + 1j * np.random.randn(*Y_noiseless.shape))
        Y_noisy = Y_noiseless + noise  # UE接收到的最终信号 (K*Nr, Nc, Q)

        # 3. UE端进行信道估计
        # Y_noisy需要是(K, Nc, Q)，因为swomp内部是按用户处理的
        # 这里假设Nr=1，所以 K*Nr = K
        A_est, G_est = swomp_channel_estimation(Y_noisy, X_pilot, At_codebook, L=L)

        # 4. BS端基于估计参数重构信道
        H_hat_system_view = reconstruct_channel(A_est, G_est, L=L)  # (K*Nr, Nc, Nt)

        # --- 路径2: 估计CSI下的波束赋形 ---
        Fopt_e, Wopt_e = get_Fopt_Wopt(H_hat_system_view, K, Nr, Ns)
        FRF_e, FBB_e = MO_AltMin(Fopt_e, NRF)

        # 对估计CSI得到的预编码器进行功率归一化
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_e @ FBB_e[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_e[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB_e[:, :, k]

        # 使用真实信道 H_sample 和基于估计信道设计的 F_e, W_e 来计算速率
        H_hat_user_view = H_hat_system_view.reshape(K, Nr, Nc, Nt).squeeze()  # (K, Nc, Nt)
        rate_e = calculate_sum_rate(H_sample_user_view, FRF_e, FBB_e, Wopt_e, snr_val, K, Ns, Nc)
        sumRateEstimated[s_idx, si] = rate_e

# --- 结果处理与绘图 ---
plt.figure(figsize=(10, 7))

final_rates_perfect = np.mean(sumRatePerfect, axis=1)
final_rates_estimated = np.mean(sumRateEstimated, axis=1)

print("\n--- MO-AltMin (perfect CSI) 平均速率 ---")
for snr, rate in zip(SNR_dB, final_rates_perfect):
    print(f"SNR: {snr} dB, Rate: {rate:.4f} bits/s/Hz")

print("\n--- MO-AltMin (estimated CSI, perfect feedback) 平均速率 ---")
for snr, rate in zip(SNR_dB, final_rates_estimated):
    print(f"SNR: {snr} dB, Rate: {rate:.4f} bits/s/Hz")

plt.plot(SNR_dB, final_rates_perfect, 'b-o', label='MO-AltMin (Perfect CSI)')
plt.plot(SNR_dB, final_rates_estimated, 'g--^', label='MO-AltMin (Estimated CSI w/ Noise)')

plt.grid(True, which='both')
plt.xlabel('SNR (dB)')
plt.ylabel('Spectral Efficiency (bits/s/Hz)')
plt.title('Performance with Perfect vs. Estimated CSI')
plt.legend()
plt.show()