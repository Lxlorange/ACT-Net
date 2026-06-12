import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from sic import sic_hybrid_precoding_ofdm
from codebook import gen_upa_cb
from swomp import estimate_swomp, recon_chan
from util import getChannel, calc_rate, gen_steer_vec
from quantization import AGQuantizer


def main():
    # --- 仿真参数 ---
    Nt = 64
    Nr = 1
    K = 2          # 固定用户数
    Ns = K
    NRF = K
    Nc = 32
    Q = 8          # 导频长度
    nt_dims = (8, 8)
    SNR_DB = 10
    snr = 10 ** (SNR_DB / 10)
    B_fixed = 36   # 固定反馈比特数
    L_vals = [1,2,3,4,5,6]   # 横坐标改成 L

    # --- 数据集加载 ---
    data = loadmat("../data/H_UPA_4.mat")
    H_data = data['H_UPA']
    H_data = H_data[:, :K, :, :]
    N_BATCH = H_data.shape[0]

    omp_ang_n = 16
    At = gen_upa_cb(nt_dims, omp_ang_n)
    Xp = (1 / np.sqrt(Q * Nt)) * (np.random.randn(Q, Nt) + 1j * np.random.randn(Q, Nt))

    # --- 用于量化器训练的随机数据 ---
    train_g = (np.random.randn(1000, K, max(L_vals)) + 1j * np.random.randn(1000, K, max(L_vals))) / np.sqrt(2)
    train_phi = np.random.uniform(-np.pi / 2, np.pi / 2, size=10000)
    train_theta = np.random.uniform(0, np.pi, size=10000)

    # --- 结果存储 ---
    rate_ideal_avg = []
    rate_sic_perfect_avg = []
    rate_sic_estimated_avg = []
    rate_sic_quantized_avg = []

    # --- 主仿真循环 ---
    for L in L_vals:
        print(f"====== 当前路径数 L = {L} ======")

        rates_ideal_digital = np.zeros(N_BATCH)
        rates_sic_perfect = np.zeros(N_BATCH)
        rates_sic_estimated = np.zeros(N_BATCH)
        rates_sic_quantized = np.zeros(N_BATCH)

        for si in range(N_BATCH):
            H_user = H_data[si, :, :, :]
            H_sys = H_user.reshape(K * Nr, Nc, Nt)

            # --- 基准 1: 理想全数字 (完美CSI) ---
            Fopt_p, Wopt_p = getChannel(H_sys, K, Nr, Ns)
            rate_p_temp = 0.0
            for k in range(Nc):
                Fk, Wk, Hk = Fopt_p[:, :, k], Wopt_p[:, :, k], H_sys[:, k, :]
                H_eff = Wk.conj().T @ Hk @ Fk
                det_term = np.eye(Ns) + (snr / Ns) * (H_eff @ H_eff.conj().T)
                sign, logdet = np.linalg.slogdet(det_term)
                if sign > 0: rate_p_temp += logdet / np.log(2)
            rates_ideal_digital[si] = np.real(rate_p_temp) / Nc

            # --- 基准 2: SIC (完美CSI) ---
            FRF_p, FBB_p = sic_hybrid_precoding_ofdm(H_sys, NRF, Ns, At)
            for k in range(Nc):
                norm_f = np.linalg.norm(FRF_p @ FBB_p[:, :, k], 'fro')
                if norm_f > 1e-9: FBB_p[:, :, k] *= (np.sqrt(Ns) / norm_f)
            rates_sic_perfect[si] = calc_rate(H_user, FRF_p, FBB_p, Wopt_p, snr, K, Ns, Nc)

            # --- 信道估计 ---
            Y_clean = np.tensordot(H_sys, Xp.T, axes=([2], [0]))
            n_var = 1 / snr
            noise = np.sqrt(n_var / 2) * (np.random.randn(*Y_clean.shape) + 1j * np.random.randn(*Y_clean.shape))
            Y = Y_clean + noise
            g, phi, theta, _ = estimate_swomp(Y.reshape(K, Nc, Q), Xp, At, L, omp_ang_n)

            # --- 基准 3: SIC (估计CSI, 无量化) ---
            A_est = np.hstack([gen_steer_vec(nt_dims, phi[l], theta[l]) for l in range(L)])
            H_hat = recon_chan(A_est, g, L)
            H_hat_sys = H_hat.reshape(K * Nr, Nc, Nt)
            _, Wopt_e = getChannel(H_hat_sys, K, Nr, Ns)
            FRF_e, FBB_e = sic_hybrid_precoding_ofdm(H_hat_sys, NRF, Ns, At)
            for k in range(Nc):
                norm_f = np.linalg.norm(FRF_e @ FBB_e[:, :, k], 'fro')
                if norm_f > 1e-9: FBB_e[:, :, k] *= (np.sqrt(Ns) / norm_f)
            rates_sic_estimated[si] = calc_rate(H_user, FRF_e, FBB_e, Wopt_e, snr, K, Ns, Nc)

            # --- 4. SIC (估计CSI + 固定B=32) ---
            quantizer = AGQuantizer(B_fixed, L, K, train_g, train_phi, train_theta)
            g_avg = np.mean(g, axis=2)
            q_data = quantizer.quantize(g_avg, phi, theta)
            g_q_avg, phi_q, theta_q = quantizer.dequantize(q_data)

            g_q = np.tile(np.expand_dims(g_q_avg, axis=2), (1, 1, Nc))
            A_q = np.hstack([gen_steer_vec(nt_dims, phi_q[l], theta_q[l]) for l in range(L)])
            H_q = recon_chan(A_q, g_q, L)
            H_q_sys = H_q.reshape(K * Nr, Nc, Nt)

            _, Wopt_q = getChannel(H_q_sys, K, Nr, Ns)
            FRF_q, FBB_q = sic_hybrid_precoding_ofdm(H_q_sys, NRF, Ns, At)
            for k in range(Nc):
                norm_f = np.linalg.norm(FRF_q @ FBB_q[:, :, k], 'fro')
                if norm_f > 1e-9: FBB_q[:, :, k] *= (np.sqrt(Ns) / norm_f)

            rates_sic_quantized[si] = calc_rate(H_user, FRF_q, FBB_q, Wopt_q, snr, K, Ns, Nc)

        # 存平均结果
        rate_ideal_avg.append(np.mean(rates_ideal_digital))
        rate_sic_perfect_avg.append(np.mean(rates_sic_perfect))
        rate_sic_estimated_avg.append(np.mean(rates_sic_estimated))
        rate_sic_quantized_avg.append(np.mean(rates_sic_quantized))

    # --- 打印结果 ---
    print("\n====== 平均结果 (固定K, B=32) ======")
    print(f"{'L':<5}{'Perfect CSI':<15}{'SIC (Perfect)':<20}{'SIC (Estimated)':<20}{'SIC+Quant (B=32)':<20}")
    for L, r1, r2, r3, r4 in zip(L_vals, rate_ideal_avg, rate_sic_perfect_avg, rate_sic_estimated_avg, rate_sic_quantized_avg):
        print(f"{L:<5}{r1:<15.4f}{r2:<20.4f}{r3:<20.4f}{r4:<20.4f}")

    # --- 绘图 ---
    plt.figure(figsize=(12, 8))
    plt.plot(L_vals, rate_ideal_avg, 'k-o', label='Ideal Full-Digital (Perfect CSI)')
    plt.plot(L_vals, rate_sic_perfect_avg, 'b--s', label='SIC (Perfect CSI)')
    plt.plot(L_vals, rate_sic_estimated_avg, 'g-.*', label='SIC (Estimated CSI)')
    plt.plot(L_vals, rate_sic_quantized_avg, 'r-d', label='SIC (Estimated CSI + Quantization, B=32)')

    plt.grid(True)
    plt.xlabel('Channel Paths (L)')
    plt.ylabel('Spectral Efficiency (bits/s/Hz)')
    plt.title(f'Performance vs. Channel Paths L (K={K}, B={B_fixed}, SNR={SNR_DB} dB)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
