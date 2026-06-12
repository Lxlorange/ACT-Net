# diffK.py
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from mano import MO_AltMin
from codebook import gen_upa_cb
from swomp import estimate_swomp, recon_chan
from util import *
from quantization import AGQuantizer

# --- Simulation Parameters ---
Nt = 64
Nr = 1
Ns = None   # will set = K in loop
NRF = None  # will set = K in loop
Nc = 32
Q = 4
nt_dims = (8, 8)
SNR_DB = 10
snr = 10 ** (SNR_DB / 10)
B_fixed = 1
L_fixed = 2
K_vals = [1, 2, 3, 4, 5, 6]

# --- Data & Codebook Loading ---
data = loadmat("../data/H_UPA_6.mat")

H_data = data['H_UPA']
N_BATCH = H_data.shape[0]

omp_ang_n = 16
At = gen_upa_cb(nt_dims, omp_ang_n)
Xp = (1 / np.sqrt(Nt)) * np.exp(1j * 2 * np.pi * np.random.rand(Q, Nt))

# --- Quantizer Training Data ---
train_g = (np.random.randn(1000, max(K_vals), L_fixed) + 1j * np.random.randn(1000, max(K_vals), L_fixed)) / np.sqrt(2)
train_phi = np.random.uniform(-np.pi / 2, np.pi / 2, size=10000)
train_theta = np.random.uniform(0, np.pi, size=10000)

# --- Result Storage ---
rates_p_vs_K = np.zeros((len(K_vals), N_BATCH))
rates_e_vs_K = np.zeros((len(K_vals), N_BATCH))
rates_q_vs_K = np.zeros((len(K_vals), N_BATCH))

# --- Main Simulation Loop ---
for si in range(N_BATCH):
    print(f"Processing sample: {si + 1}/{N_BATCH}")

    for k_idx, K in enumerate(K_vals):
        Ns = K
        NRF = K

        H_user = H_data[si, :K, :, :]     # 取前 K 个用户
        H_sys = H_user.reshape(K * Nr, Nc, Nt)

        # # --- Benchmark 1: Perfect CSI ---
        # Fopt, Wopt = getChannel(H_sys, K, Nr, Ns)
        # FRF, FBB = MO_AltMin(Fopt, NRF)
        # for k in range(Nc):
        #     norm_f = np.linalg.norm(FRF @ FBB[:, :, k], 'fro')
        #     if norm_f > 1e-9:
        #         FBB[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB[:, :, k]
        # rates_p_vs_K[k_idx, si] = calc_rate(H_user, FRF, FBB, Wopt, snr, K, Ns, Nc)

        # --- Benchmark 2: Estimated CSI (No Quantization) ---
        Y_clean = np.tensordot(H_sys, Xp.T, axes=([2], [0]))
        n_var = 1 / snr
        noise = np.sqrt(n_var / 2) * (np.random.randn(*Y_clean.shape) + 1j * np.random.randn(*Y_clean.shape))
        Y = Y_clean + noise
        g, phi, theta, path_idx = estimate_swomp(Y, Xp, At, L_fixed, omp_ang_n)
        A_est = np.hstack([gen_steer_vec(nt_dims, phi[l], theta[l]) for l in range(L_fixed)])
        H_hat = recon_chan(A_est, g, L_fixed)
        Fopt_e, Wopt_e = getChannel(H_hat, K, Nr, Ns)
        FRF_e, FBB_e = MO_AltMin(Fopt_e, NRF)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_e @ FBB_e[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_e[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB_e[:, :, k]
        rates_e_vs_K[k_idx, si] = calc_rate(H_user, FRF_e, FBB_e, Wopt_e, snr, K, Ns, Nc)

        # --- Path 3: Estimated CSI with Limited Feedback ---
        qtz = AGQuantizer(B_fixed, L_fixed, K, train_g[:, :K, :], train_phi, train_theta)
        g_avg = np.mean(g, axis=2)
        q_data = qtz.quantize(g_avg, phi, theta)
        g_q_avg, phi_q, theta_q = qtz.dequantize(q_data)
        g_q = np.tile(np.expand_dims(g_q_avg, axis=2), (1, 1, Nc))
        A_q = np.hstack([gen_steer_vec(nt_dims, phi_q[l], theta_q[l]) for l in range(L_fixed)])
        H_q = recon_chan(A_q, g_q, L_fixed)
        Fopt_q, Wopt_q = getChannel(H_q, K, Nr, Ns)
        FRF_q, FBB_q = MO_AltMin(Fopt_q, NRF)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_q @ FBB_q[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_q[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB_q[:, :, k]
        rates_q_vs_K[k_idx, si] = calc_rate(H_user, FRF_q, FBB_q, Wopt_q, snr, K, Ns, Nc)

# --- Results & Plotting ---
plt.figure(figsize=(10, 7))

rate_p_avg = np.mean(rates_p_vs_K, axis=1)
rate_e_avg = np.mean(rates_e_vs_K, axis=1)
rate_q_avg = np.mean(rates_q_vs_K, axis=1)

print(f"\n--- Fixed SNR = {SNR_DB} dB, B = {B_fixed}, L = {L_fixed} ---")
for k, r_p, r_e, r_q in zip(K_vals, rate_p_avg, rate_e_avg, rate_q_avg):
    print(f"K = {k}: Perfect={r_p:.4f}, EstNoQ={r_e:.4f}, Quantized={r_q:.4f}")

# plt.plot(K_vals, rate_p_avg, 'b-o', label='Perfect CSI')
plt.plot(K_vals, rate_e_avg, 'g-^', label='Estimated CSI (No Quantization)')
plt.plot(K_vals, rate_q_avg, 'r-s', label=f'Estimated CSI with Limited Feedback (B={B_fixed})')

plt.grid(True, which='both')
plt.xlabel('Number of Users (K)')
plt.ylabel('Spectral Efficiency (bits/s/Hz)')
plt.title(f'Performance vs. K at SNR = {SNR_DB} dB, B = {B_fixed}, L = {L_fixed}')
plt.legend()
plt.xticks(K_vals, labels=K_vals)
plt.show()
