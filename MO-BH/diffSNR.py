import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

# 导入您项目中的自定义函数
from mano import MO_AltMin
from codebook import gen_upa_cb
from swomp import *
from util import *
from quantization import AGQuantizer

# --- 仿真参数设置 ---
Nt = 64
Nr = 1
K = 2  # 用户数
Ns = K  # 每个用户一个数据流，总数据流数
NRF = K  # 射频链数量
Nc = 32  # 子载波数
L = 2  # 信道稀疏度 (路径数)
Q = 4 #导频长度
B = 64

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
At_codebook = gen_upa_cb(Nt_params, angle_samples)
random_phases = 2 * np.pi * np.random.rand(Q, Nt)
N_atoms = At_codebook.shape[1]
X_pilot = (1 / np.sqrt(Nt)) * np.exp(1j * random_phases)
print(f"码本生成完毕，维度: {At_codebook.shape}")
print(f"导频生成完毕，维度: {X_pilot.shape}")

print("正在生成用于训练量化器的增益样本...")
training_gain_samples = []
for i in range(min(10, BATCH_SIZE)): # 取10个样本进行训练
    _, G_est_sample, _ = estimate_swomp(
        np.tensordot(H_dataset[i].reshape(K*Nr, Nc, Nt), X_pilot.T, axes=([2],[0])),
        X_pilot, At_codebook, L=L
    )
    training_gain_samples.append(G_est_sample)
training_gains = np.concatenate(training_gain_samples, axis=0)
# 创建量化器实例，它会自动完成比特分配和码本训练
quantizer = AGQuantizer(B, L, K, Nc, N_atoms, training_gains)


sumRatePerfect = np.zeros((len(SNR_dB), BATCH_SIZE))
sumRateEstimated = np.zeros((len(SNR_dB), BATCH_SIZE))
sumRateFeedback = np.zeros((len(SNR_dB), BATCH_SIZE))

for si in range(BATCH_SIZE):
    print(f"正在处理样本: {si + 1}/{BATCH_SIZE}")
    H_sample_user_view = H_dataset[si, :, :, :]  # (K, Nc, Nt)
    # 将信道矩阵转换为 (K*Nr, Nc, Nt) 的形式，以便 get_Fopt_Wopt 处理
    H_sample_system_view = H_sample_user_view.reshape(K * Nr, Nc, Nt)

    # --- 路径1: 完美CSI下的波束赋形 ---
    Fopt_p, Wopt_p = getChannel(H_sample_system_view, K, Nr, Ns)
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
        sumRatePerfect[s_idx, si] = calc_rate(H_sample_user_view, FRF_p, FBB_p, Wopt_p, snr_val, K, Ns, Nc)

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
        A_est, G_est, path_indices = estimate_swomp(Y_noisy, X_pilot, At_codebook, L=L)
        # 4. BS端基于估计参数重构信道
        H_hat_system_view = recon_chan(A_est, G_est, L=L)  # (K*Nr, Nc, Nt)

        # --- 路径2: 估计CSI下的波束赋形 ---
        Fopt_e, Wopt_e = getChannel(H_hat_system_view, K, Nr, Ns)
        FRF_e, FBB_e = MO_AltMin(Fopt_e, NRF)

        # 对估计CSI得到的预编码器进行功率归一化
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_e @ FBB_e[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_e[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB_e[:, :, k]
        sumRateEstimated[s_idx, si] = calc_rate(
        H_sample_user_view, FRF_e, FBB_e, Wopt_e, snr_val, K, Ns, Nc)

        # --- 路径3: 估计CSI + 有限反馈 ---
        # a. UE端量化
        q_indices, q_gains_real, q_gains_imag = quantizer.quantize(path_indices, G_est)

        # b. BS端反量化
        dq_indices, G_quant = quantizer.dequantize(q_indices, q_gains_real, q_gains_imag)

        # c. BS端根据有损参数重构信道
        A_quant = At_codebook[:, dq_indices]  # 从码本中恢复方向向量
        H_quant_system_view = recon_chan(A_quant, G_quant, L=L)

        # d. BS端基于量化后的信道进行波束赋形
        Fopt_q, Wopt_q = getChannel(H_quant_system_view, K, Nr, Ns)
        FRF_q, FBB_q = MO_AltMin(Fopt_q, NRF)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_q @ FBB_q[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_q[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB_q[:, :, k]

        # e. 使用真实信道和量化设计的波束赋形计算速率
        sumRateFeedback[s_idx, si] = calc_rate(
            H_sample_user_view, FRF_q, FBB_q, Wopt_q, snr_val, K, Ns, Nc)

# --- 结果处理与绘图 ---
plt.figure(figsize=(10, 7))

final_rates_perfect = np.mean(sumRatePerfect, axis=1)
final_rates_estimated = np.mean(sumRateEstimated, axis=1)
final_rates_feedback = np.mean(sumRateFeedback, axis=1)  # 新增

print("\n--- 完美CSI 平均速率 ---")
for snr, rate in zip(SNR_dB, final_rates_perfect): print(f"SNR: {snr} dB, Rate: {rate:.4f}")
print("\n--- 估计CSI+完美反馈 平均速率 ---")
for snr, rate in zip(SNR_dB, final_rates_estimated): print(f"SNR: {snr} dB, Rate: {rate:.4f}")
print("\n--- 估计CSI+有限反馈 平均速率 ---")
for snr, rate in zip(SNR_dB, final_rates_feedback): print(f"SNR: {snr} dB, Rate: {rate:.4f}")

plt.plot(SNR_dB, final_rates_perfect, 'b-o', label='Perfect CSI')
plt.plot(SNR_dB, final_rates_estimated, 'g--^', label='Estimated CSI (Perfect Feedback)')
plt.plot(SNR_dB, final_rates_feedback, 'r-.s', label=f'Estimated CSI ({B}-bit Feedback)')  # 新增

plt.grid(True, which='both')
plt.xlabel('SNR (dB)')
plt.ylabel('Spectral Efficiency (bits/s/Hz)')
plt.title('Performance Comparison with Limited Feedback')
plt.legend()
plt.show()