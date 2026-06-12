# diffL.py
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from mano import MO_AltMin
from codebook import gen_upa_cb
from swomp import estimate_swomp, recon_chan
from util import *
from quantization import AGQuantizer  # IdxGainQuantizer 也能用

# --- Simulation Parameters ---
Nt = 64
Nr = 1
K = 2
Ns = K
NRF = K
Nc = 32
Q = 4
nt_dims = (8, 8)
SNR_DB = 10
snr = 10 ** (SNR_DB / 10)
B_fixed = 32
L_vals = [1, 2, 3, 4, 5, 6]   # <-- L 变化

# --- Data & Codebook Loading ---
data = loadmat("../data/H_UPA_4.mat")

H_data = data['H_UPA']
N_BATCH = H_data.shape[0]
H_data = H_data[:, :K, :, :]

omp_ang_n = 16
At = gen_upa_cb(nt_dims, omp_ang_n)
Xp = (1 / np.sqrt(Nt)) * np.exp(1j * 2 * np.pi * np.random.rand(Q, Nt))

# --- Quantizer Training Data ---
train_g = (np.random.randn(1000, K, max(L_vals)) + 1j * np.random.randn(1000, K, max(L_vals))) / np.sqrt(2)
train_phi = np.random.uniform(-np.pi / 2, np.pi / 2, size=10000)
train_theta = np.random.uniform(0, np.pi, size=10000)

# --- Result Storage ---
rates_p = np.zeros(N_BATCH)                 # Perfect CSI
rates_e_vs_L = np.zeros((len(L_vals), N_BATCH))   # Estimated CSI, no quantization
rates_q_vs_L = np.zeros((len(L_vals), N_BATCH))   # Estimated CSI + Quantization


# --- Main Simulation Loop ---
for si in range(N_BATCH):
    print(f"Processing sample: {si + 1}/{N_BATCH}")
    H_user = H_data[si, :, :, :]
    H_sys = H_user.reshape(K * Nr, Nc, Nt)

    # --- Benchmark 1: Perfect CSI ---
    Fopt, Wopt = getChannel(H_sys, K, Nr, Ns)
    FRF, FBB = MO_AltMin(Fopt, NRF)
    for k in range(Nc):
        norm_f = np.linalg.norm(FRF @ FBB[:, :, k], 'fro')
        if norm_f > 1e-9:
            FBB[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB[:, :, k]
    rates_p[si] = calc_rate(H_user, FRF, FBB, Wopt, snr, K, Ns, Nc)

    # --- Loop over L values ---
    for l_idx, L in enumerate(L_vals):
        # --- 信道估计 ---
        Y_clean = np.tensordot(H_sys, Xp.T, axes=([2], [0]))
        n_var = 1 / snr
        noise = np.sqrt(n_var / 2) * (np.random.randn(*Y_clean.shape) + 1j * np.random.randn(*Y_clean.shape))
        Y = Y_clean + noise
        g, phi, theta, path_idx = estimate_swomp(Y, Xp, At, L, omp_ang_n)

        # --- 无量化 ---
        A_est = np.hstack([gen_steer_vec(nt_dims, phi[l], theta[l]) for l in range(L)])
        H_hat = recon_chan(A_est, g, L)
        Fopt_e, Wopt_e = getChannel(H_hat, K, Nr, Ns)
        FRF_e, FBB_e = MO_AltMin(Fopt_e, NRF)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_e @ FBB_e[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_e[:, :, k] *= np.sqrt(Ns) / norm_f
        rates_e_vs_L[l_idx, si] = calc_rate(H_user, FRF_e, FBB_e, Wopt_e, snr, K, Ns, Nc)

        # --- 有量化 ---
        qtz = AGQuantizer(B_fixed, L, K, train_g[:, :, :L], train_phi, train_theta)
        g_avg = np.mean(g, axis=2)
        q_data = qtz.quantize(g_avg, phi, theta)
        g_q_avg, phi_q, theta_q = qtz.dequantize(q_data)
        g_q = np.tile(np.expand_dims(g_q_avg, axis=2), (1, 1, Nc))
        A_q = np.hstack([gen_steer_vec(nt_dims, phi_q[l], theta_q[l]) for l in range(L)])
        H_q = recon_chan(A_q, g_q, L)
        Fopt_q, Wopt_q = getChannel(H_q, K, Nr, Ns)
        FRF_q, FBB_q = MO_AltMin(Fopt_q, NRF)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_q @ FBB_q[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_q[:, :, k] *= np.sqrt(Ns) / norm_f
        rates_q_vs_L[l_idx, si] = calc_rate(H_user, FRF_q, FBB_q, Wopt_q, snr, K, Ns, Nc)

# --- Results & Plotting ---
rate_p_avg = np.mean(rates_p)
rate_e_avg = np.mean(rates_e_vs_L, axis=1)
rate_q_avg = np.mean(rates_q_vs_L, axis=1)

print(rate_p_avg)
print(rate_e_avg)
print(rate_q_avg)
plt.figure(figsize=(10,7))
plt.axhline(y=rate_p_avg, color='b', linestyle='-', label='Perfect CSI (Upper Bound)')
plt.plot(L_vals, rate_e_avg, 'g--o', label='Estimated CSI (No Quantization)')
plt.plot(L_vals, rate_q_avg, 'r-.s', label=f'Estimated CSI + Limited Feedback (B={B_fixed})')

plt.grid(True, which='both')
plt.xlabel('Number of Paths (L)')
plt.ylabel('Spectral Efficiency (bits/s/Hz)')
plt.title(f'Performance vs. L at SNR={SNR_DB} dB, B={B_fixed}')
plt.legend()
plt.xticks(L_vals, labels=L_vals)
plt.show()

