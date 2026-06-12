import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# 导入您自己的模块
from sic import sic_hybrid_precoding_ofdm
from codebook import gen_upa_cb
from swomp import estimate_swomp, recon_chan
from util import getChannel, calc_rate, gen_steer_vec
from quantization import AGQuantizer


def run_simulation(K, SNR_DB=10, L=2):
    """运行一次仿真，返回平均速率结果"""
    Nt = 64
    Nr = 1
    Ns = K
    NRF = K
    Nc = 32
    Q = 8
    nt_dims = (8, 8)
    snr = 10 ** (SNR_DB / 10)
    B_vals = [32]

    # 数据加载
    try:
        data = loadmat("../data/H_UPA_6.mat")
    except FileNotFoundError:
        print("错误: 数据集 'H_UPA_6.mat' 未在 '../data/' 目录下找到。")
        return None

    H_data = data['H_UPA']
    if K > H_data.shape[1]:
        print(f"警告: 数据集中最多支持 {H_data.shape[1]} 用户, 跳过 K={K}")
        return None

    H_data = H_data[:, :K, :, :]
    N_BATCH = H_data.shape[0]

    omp_ang_n = 16
    At = gen_upa_cb(nt_dims, omp_ang_n)
    Xp = (1 / np.sqrt(Q * Nt)) * (np.random.randn(Q, Nt) + 1j * np.random.randn(Q, Nt))

    train_g = (np.random.randn(1000, K, L) + 1j * np.random.randn(1000, K, L)) / np.sqrt(2)
    train_phi = np.random.uniform(-np.pi / 2, np.pi / 2, size=10000)
    train_theta = np.random.uniform(0, np.pi, size=10000)

    # 存储结果
    rates_ideal_digital = np.zeros(N_BATCH)
    rates_sic_perfect = np.zeros(N_BATCH)
    rates_sic_estimated = np.zeros(N_BATCH)
    rates_sic_quantized_vs_B = np.zeros((len(B_vals), N_BATCH))

    for si in range(N_BATCH):
        H_user = H_data[si, :, :, :]
        H_sys = H_user.reshape(K * Nr, Nc, Nt)

        # 1. 理想全数字
        Fopt_p, Wopt_p = getChannel(H_sys, K, Nr, Ns)
        rate_p_temp = 0.0
        for k in range(Nc):
            Fk, Wk, Hk = Fopt_p[:, :, k], Wopt_p[:, :, k], H_sys[:, k, :]
            H_eff = Wk.conj().T @ Hk @ Fk
            det_term = np.eye(Ns) + (snr / Ns) * (H_eff @ H_eff.conj().T)
            sign, logdet = np.linalg.slogdet(det_term)
            if sign > 0:
                rate_p_temp += logdet / np.log(2)
        rates_ideal_digital[si] = np.real(rate_p_temp) / Nc

        # 2. SIC (完美CSI)
        FRF_p, FBB_p = sic_hybrid_precoding_ofdm(H_sys, NRF, Ns, At)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_p @ FBB_p[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_p[:, :, k] *= (np.sqrt(Ns) / norm_f)
        rates_sic_perfect[si] = calc_rate(H_user, FRF_p, FBB_p, Wopt_p, snr, K, Ns, Nc)

        # 3. 信道估计
        Y_clean = np.tensordot(H_sys, Xp.T, axes=([2], [0]))
        n_var = 1 / snr
        noise = np.sqrt(n_var / 2) * (np.random.randn(*Y_clean.shape) + 1j * np.random.randn(*Y_clean.shape))
        Y = Y_clean + noise
        g, phi, theta, _ = estimate_swomp(Y.reshape(K, Nc, Q), Xp, At, L, omp_ang_n)

        A_est = np.hstack([gen_steer_vec(nt_dims, phi[l], theta[l]) for l in range(L)])
        H_hat = recon_chan(A_est, g, L)
        H_hat_sys = H_hat.reshape(K * Nr, Nc, Nt)
        _, Wopt_e = getChannel(H_hat_sys, K, Nr, Ns)
        FRF_e, FBB_e = sic_hybrid_precoding_ofdm(H_hat_sys, NRF, Ns, At)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_e @ FBB_e[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_e[:, :, k] *= (np.sqrt(Ns) / norm_f)
        rates_sic_estimated[si] = calc_rate(H_user, FRF_e, FBB_e, Wopt_e, snr, K, Ns, Nc)

        # 4. SIC + 有限反馈
        for b_idx, b_val in enumerate(B_vals):
            quantizer = AGQuantizer(b_val, L, K, train_g, train_phi, train_theta)
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
                if norm_f > 1e-9:
                    FBB_q[:, :, k] *= (np.sqrt(Ns) / norm_f)

            rates_sic_quantized_vs_B[b_idx, si] = calc_rate(H_user, FRF_q, FBB_q, Wopt_q, snr, K, Ns, Nc)

    # 平均结果
    return {
        "K": K,
        "rate_ideal": np.mean(rates_ideal_digital),
        "rate_sic_perfect": np.mean(rates_sic_perfect),
        "rate_sic_estimated": np.mean(rates_sic_estimated),
        "rate_sic_quantized": np.mean(rates_sic_quantized_vs_B, axis=1),
        "B_vals": B_vals
    }


if __name__ == "__main__":
    all_results = []
    for K in range(1, 7):
        print(f"\n===== 开始仿真 K={K} =====")
        result = run_simulation(K)
        if result is not None:
            all_results.append(result)

    # --- 打印统一结果 ---
    print("\n" + "=" * 80)
    print("最终仿真结果 (SNR=10 dB, L=2)")
    print("=" * 80)
    for res in all_results:
        K = res["K"]
        print(f"\n>>> K = {K}")
        print(f"  - 理想全数字 (Perfect CSI) : {res['rate_ideal']:.4f}")
        print(f"  - SIC (Perfect CSI)       : {res['rate_sic_perfect']:.4f}")
        print(f"  - SIC (Estimated CSI)     : {res['rate_sic_estimated']:.4f}")
        print("  - SIC + Quantization:")
        for b, r in zip(res["B_vals"], res["rate_sic_quantized"]):
            print(f"      B={b:<3} : {r:.4f}")

    # --- 绘制随 K 变化的平均速率 (固定 B=32) ---
    Ks = [res["K"] for res in all_results]

    plt.figure(figsize=(10, 6))
    # plt.plot(Ks, [res["rate_ideal"] for res in all_results], 'k-o', label="Ideal Digital (Perfect CSI)")
    # plt.plot(Ks, [res["rate_sic_perfect"] for res in all_results], 'b--s', label="SIC (Perfect CSI)")
    plt.plot(Ks, [res["rate_sic_estimated"] for res in all_results], 'g-.*', label="SIC (Estimated CSI)")

    # 只取 B=32 的点
    B_fixed = 32
    b_idx = all_results[0]["B_vals"].index(B_fixed)
    plt.plot(Ks, [res["rate_sic_quantized"][b_idx] for res in all_results],
             'r-d', label=f"SIC (Estimated CSI + Quantization, B={B_fixed})")

    plt.xlabel("Number of Users (K)")
    plt.ylabel("Average Spectral Efficiency (bits/s/Hz)")
    plt.title(f"Performance vs K (SNR=10 dB, L=2, B={B_fixed})")
    plt.grid(True)
    plt.legend()
    plt.show()

