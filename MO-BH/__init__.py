import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mano import *
Nt = 64
Nr = 1
K = 2
Ns = K  # num of stream = num of RF
NRF = K
# Np = 2  # clusters * rays = 1 * 2 = 2

Nc = 32

SNR_dB = np.arange(5, 16, 5)
SNR_lin = 10**(SNR_dB / 10)

data = loadmat("../data/H_UPA_4.mat")
H_dataset = data['H_UPA']

BATCH_SIZE = H_dataset.shape[0]
print(f"H_UPA, Shape: {H_dataset.shape}")

sumRateMO = np.zeros((len(SNR_dB),BATCH_SIZE))
H_dataset = H_dataset[:,:K,:,:] #保留K个用户

for si in range(BATCH_SIZE):
    H_sample = H_dataset[si,:,:,:]
    Fopt = np.zeros((Nt,Ns,Nc),dtype=np.complex128)
    Wopt = np.zeros((Nr*K,Ns,Nc),dtype=np.complex128)

    for k in range(Nc):
        # 构造当前子载波的复合信道矩阵 (K * Nr, Nt)
        # H_sample[:, k, :] 的 shape 是 (2, 64)，因为 Nr=1，所以 reshape 后仍是 (2, 64)
        H_k_composite = H_sample[:, k, :].reshape(K * Nr, Nt)

        U, S, Vh = np.linalg.svd(H_k_composite, full_matrices=False)
        Fopt[:, :, k] = Vh.conj().T[:, :Ns]
        Wopt[:, :, k] = U[:, :Ns]

    FRFM, FBBM = MO_AltMin(Fopt, NRF)

    #功率归一化
    for k in range(Nc):
        # 计算复合预编码器的F范数
        norm_factor = np.linalg.norm(FRFM @ FBBM[:, :, k], 'fro')
        # 归一化以满足功率约束 ||F_RF * F_BB||^2 = Ns
        if norm_factor > 1e-9:
            FBBM[:, :, k] = np.sqrt(Ns) * FBBM[:, :, k] / norm_factor

    WRFM, WBBM = MO_AltMin(Wopt, NRF)

    # 计算频谱效率
    for s_idx, snr_val in enumerate(SNR_lin):
        total_rate_per_snr = 0.0
        for k_sub in range(Nc):
            # 获取当前子载波的信道和预编码/合并矩阵
            H_k = H_sample[:, k_sub, :].reshape(K * Nr, Nt)
            F_k = FRFM @ FBBM[:, :, k_sub]
            W_k = WRFM @ WBBM[:, :, k_sub]

            # MIMO系统容量公式
            # R = log2(det(I + (SNR/Ns) * inv(W'W) * W' * H * F * F' * H' * W))
            # 由于W是理想的MMSE合并器的近似，inv(W'W)*W' 可以用 pinv(W) 代替
            term_inside_det = np.eye(Ns, dtype=np.complex128) + (snr_val / Ns) * \
                              np.linalg.pinv(W_k) @ H_k @ F_k @ F_k.conj().T @ H_k.conj().T @ W_k

            # 使用 slogdet 计算 log(det(...)) 以保证数值稳定性
            sign, logdet = np.linalg.slogdet(term_inside_det)

            # 确保行列式为正
            if sign > 0:
                # np.log 是自然对数，需要除以 np.log(2) 转换为以2为底
                total_rate_per_snr += logdet / np.log(2)

        # 将当前SNR下的总速率（已累加所有子载波）除以子载波数，存入结果矩阵
        sumRateMO[s_idx, si] = total_rate_per_snr / Nc


plt.figure(figsize=(10, 7))
# 对所有样本(BATCH_SIZE)的结果求平均
final_rates = np.mean(sumRateMO, axis=1)
print(final_rates)
plt.plot(SNR_dB, final_rates, 'b-o', label='MO-AltMin')

plt.grid(True, which='both')
plt.xlabel('SNR (dB)')
plt.ylabel('Spectral Efficiency (bits/s/Hz)')
plt.title('MO-AltMin Performance in DL Context')
plt.legend()
plt.show()


