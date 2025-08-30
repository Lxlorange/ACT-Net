# _*_ coding : utf-8 _*_
# @Time :  17:51
# @Author : Lxl
# @File ： diffB
# @ProjectName : OFDM
# __init__.py

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

# 导入您项目中的自定义函数
from mano import MO_AltMin
from codebook import generate_upa_codebook
from swomp import swomp_channel_estimation, reconstruct_channel
from util import *
from quantization import FeedbackQuantizer

# --- 仿真参数设置 ---
Nt = 64
Nr = 1
K_user = 2
Ns = K_user
NRF = K_user
Nc = 32
L = 2
Q = 8

# MODIFIED: 固定SNR，设置B的范围
FIXED_SNR_DB = 10  # 在这个信噪比下进行测试
snr_val = 10 ** (FIXED_SNR_DB / 10)
B_values = [8, 16, 20, 30, 40, 50, 60, 70]  # 要测试的总比特数范围

# --- 数据与码本加载 ---
try:
    data = loadmat("../data/H_UPA_4.mat")
except FileNotFoundError:
    print("错误：找不到数据文件 'data/H_UPA_4.mat'。请确保文件路径正确。")
    exit()

H_dataset = data['H_UPA']
BATCH_SIZE = H_dataset.shape[0]
print(f"加载数据集 H_UPA, 维度: {H_dataset.shape}")
H_dataset = H_dataset[:, :K_user, :, :]

Nt_params = (8, 8)
angle_samples = 32
At_codebook = generate_upa_codebook(Nt_params, angle_samples)
N_atoms = At_codebook.shape[1]

random_phases = 2 * np.pi * np.random.rand(Q, Nt)
X_pilot = (1 / np.sqrt(Nt)) * np.exp(1j * random_phases)

# --- 初始化量化器训练数据 ---
print("正在生成用于训练量化器的平均增益样本...")
training_gain_avg_samples = []
for i in range(min(10, BATCH_SIZE)):
    _, G_est_sample, _ = swomp_channel_estimation(
        np.tensordot(H_dataset[i].reshape(K_user*Nr, Nc, Nt), X_pilot.T, axes=([2],[0])), # Corrected function name
        X_pilot, At_codebook, L=L
    )
    # 计算平均增益
    training_gain_avg_samples.append(np.mean(G_est_sample, axis=2))

training_gains_avg = np.concatenate(training_gain_avg_samples, axis=0)

# --- 结果存储矩阵 ---
# MODIFIED: 结果矩阵现在以B的索引作为维度
sumRatePerfect_at_SNR = np.zeros(BATCH_SIZE)
sumRateEstimated_at_SNR = np.zeros(BATCH_SIZE)
sumRateFeedback_vs_B = np.zeros((len(B_values), BATCH_SIZE))

# --- 主仿真循环 ---
for si in range(BATCH_SIZE):
    print(f"正在处理样本: {si + 1}/{BATCH_SIZE}")
    H_sample_user_view = H_dataset[si, :, :, :]
    H_sample_system_view = H_sample_user_view.reshape(K_user * Nr, Nc, Nt)

    # --- 计算基准性能 (完美CSI 和 估计CSI下的完美反馈) ---
    # 由于SNR固定，这两个基准值对每个样本只需计算一次

    # 路径1: 完美CSI
    Fopt_p, Wopt_p = get_Fopt_Wopt(H_sample_system_view, K_user, Nr, Ns)
    FRF_p, FBB_p = MO_AltMin(Fopt_p, NRF)
    for k in range(Nc):
        norm_f = np.linalg.norm(FRF_p @ FBB_p[:, :, k], 'fro')
        if norm_f > 1e-9:
            FBB_p[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB_p[:, :, k]
    sumRatePerfect_at_SNR[si] = calculate_sum_rate(H_sample_user_view, FRF_p, FBB_p, Wopt_p, snr_val, K_user, Ns, Nc)

    # 路径2: 估计CSI (完美反馈)
    Y_noiseless = np.tensordot(H_sample_system_view, X_pilot.T, axes=([2], [0]))
    noise_variance = 1 / snr_val
    noise = np.sqrt(noise_variance / 2) * (
                np.random.randn(*Y_noiseless.shape) + 1j * np.random.randn(*Y_noiseless.shape))
    Y_noisy = Y_noiseless + noise
    A_est, G_est, path_indices = swomp_channel_estimation(Y_noisy, X_pilot, At_codebook, L=L)
    H_hat_system_view = reconstruct_channel(A_est, G_est, L=L)
    Fopt_e, Wopt_e = get_Fopt_Wopt(H_hat_system_view, K_user, Nr, Ns)
    FRF_e, FBB_e = MO_AltMin(Fopt_e, NRF)
    for k in range(Nc):
        norm_f = np.linalg.norm(FRF_e @ FBB_e[:, :, k], 'fro')
        if norm_f > 1e-9:
            FBB_e[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB_e[:, :, k]
    sumRateEstimated_at_SNR[si] = calculate_sum_rate(H_sample_user_view, FRF_e, FBB_e, Wopt_e, snr_val, K_user, Ns, Nc)

    # NEW: 循环遍历不同的B值
    for b_idx, b_val in enumerate(B_values):
        # NEW: 用平均增益数据来初始化量化器
        quantizer = FeedbackQuantizer(b_val, L, K_user, N_atoms, training_gains_avg)

        # --- 路径3: 估计CSI + 有限反馈 ---
        # a. UE端量化 (内部自动计算平均值)
        q_indices, q_gains_real, q_gains_imag = quantizer.quantize(path_indices, G_est)

        # b. BS端反量化 (需要传入Nc来扩展维度)
        dq_indices, G_quant = quantizer.dequantize(q_indices, q_gains_real, q_gains_imag, Nc)

        # c. BS端根据有损参数重构信道
        A_quant = At_codebook[:, dq_indices]
        H_quant_system_view = reconstruct_channel(A_quant, G_quant, L=L)

        # d. BS端基于量化后的信道进行波束赋形
        Fopt_q, Wopt_q = get_Fopt_Wopt(H_quant_system_view, K_user, Nr, Ns)
        FRF_q, FBB_q = MO_AltMin(Fopt_q, NRF)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_q @ FBB_q[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_q[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB_q[:, :, k]

        # e. 计算速率并存储
        sumRateFeedback_vs_B[b_idx, si] = calculate_sum_rate(H_sample_user_view, FRF_q, FBB_q, Wopt_q, snr_val, K_user,
                                                             Ns, Nc)

# --- 结果处理与绘图 ---
# MODIFIED: 修改绘图逻辑
plt.figure(figsize=(10, 7))

# 计算三个场景的平均性能
final_rate_perfect = np.mean(sumRatePerfect_at_SNR)
final_rate_estimated = np.mean(sumRateEstimated_at_SNR)
final_rates_feedback = np.mean(sumRateFeedback_vs_B, axis=1)

print(f"\n--- 固定 SNR = {FIXED_SNR_DB} dB ---")
print(f"Perfect CSI 性能: {final_rate_perfect:.4f}")
print(f"Estimated CSI (Perfect Feedback) 性能: {final_rate_estimated:.4f}")
print("\n--- 有限反馈性能 vs. B ---")
for b, rate in zip(B_values, final_rates_feedback):
    print(f"B = {b} bits, Rate: {rate:.4f}")

# 绘制两条水平线作为性能基准
plt.axhline(y=final_rate_perfect, color='b', linestyle='-', label='Perfect CSI (Upper Bound)')
plt.axhline(y=final_rate_estimated, color='g', linestyle='--', label='Estimated CSI (No Quantization)')

# 绘制有限反馈性能曲线
plt.plot(B_values, final_rates_feedback, 'r-.s', label='Estimated CSI with Limited Feedback')

# 设置对数X轴，以更好地观察比特数加倍的效果
plt.xscale('log', base=2)
plt.grid(True, which='both')
plt.xlabel('Total Feedback Bits (B)')
plt.ylabel('Spectral Efficiency (bits/s/Hz)')
plt.title(f'Performance vs. Feedback Bits at SNR = {FIXED_SNR_DB} dB')
plt.legend()
plt.xticks(B_values, labels=B_values)  # 确保X轴刻度清晰
plt.show()