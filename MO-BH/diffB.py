# diffB.py
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

from mano import MO_AltMin
from codebook import gen_upa_cb
from swomp import estimate_swomp, recon_chan
from util import *
from quantization import AGQuantizer, IdxGainQuantizer

# --- Simulation Parameters ---
Nt = 64
Nr = 1
K = 2
Ns = K
NRF = K
Nc = 32
L = 2
Q = 2
nt_dims = (8, 8)
SNR_DB = 10
snr = 10 ** (SNR_DB / 10)
B_vals = [1, 3, 16, 24, 32, 48, 64]

# --- Data & Codebook Loading ---
data = loadmat("../data/H_UPA_4.mat")

H_data = data['H_UPA']
N_BATCH = H_data.shape[0]
H_data = H_data[:, :K, :, :]

omp_ang_n = 16
At = gen_upa_cb(nt_dims, omp_ang_n)
Xp = (1 / np.sqrt(Nt)) * np.exp(1j * 2 * np.pi * np.random.rand(Q, Nt))

# --- Quantizer Training Data ---
train_g = (np.random.randn(1000, K, L) + 1j * np.random.randn(1000, K, L)) / np.sqrt(2)
train_phi = np.random.uniform(-np.pi / 2, np.pi / 2, size=10000)
train_theta = np.random.uniform(0, np.pi, size=10000)

# --- Result Storage ---
rates_p = np.zeros(N_BATCH)
rates_e = np.zeros(N_BATCH)
rates_q_vs_B = np.zeros((len(B_vals), N_BATCH))

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

    # --- Benchmark 2: Estimated CSI (No Quantization) ---
    Y_clean = np.tensordot(H_sys, Xp.T, axes=([2], [0]))
    n_var = 1 / snr
    noise = np.sqrt(n_var / 2) * (np.random.randn(*Y_clean.shape) + 1j * np.random.randn(*Y_clean.shape))
    Y = Y_clean + noise
    g, phi, theta, path_idx = estimate_swomp(Y, Xp, At, L, omp_ang_n)
    A_est = np.hstack([gen_steer_vec(nt_dims, phi[l], theta[l]) for l in range(L)])
    H_hat = recon_chan(A_est, g, L)
    Fopt, Wopt = getChannel(H_hat, K, Nr, Ns)
    FRF, FBB = MO_AltMin(Fopt, NRF)
    for k in range(Nc):
        norm_f = np.linalg.norm(FRF @ FBB[:, :, k], 'fro')
        if norm_f > 1e-9:
            FBB[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB[:, :, k]
    rates_e[si] = calc_rate(H_user, FRF, FBB, Wopt, snr, K, Ns, Nc)

    # --- Path 3: Estimated CSI with Limited Feedback ---
    for b_idx, b_val in enumerate(B_vals):

        # AG
        qtz = AGQuantizer(b_val, L, K, train_g, train_phi, train_theta)
        g_avg = np.mean(g, axis=2)
        q_data = qtz.quantize(g_avg, phi, theta)
        g_q_avg, phi_q, theta_q = qtz.dequantize(q_data)
        g_q = np.tile(np.expand_dims(g_q_avg, axis=2), (1, 1, Nc))
        A_q = np.hstack([gen_steer_vec(nt_dims, phi_q[l], theta_q[l]) for l in range(L)])

        # Idx
        # n_atoms = omp_ang_n * omp_ang_n
        # qtz = IdxGainQuantizer(b_val,L,K,n_atoms,train_g)
        # q_data = qtz.quantize(path_idx,g)
        # path_idx_q, g_q = qtz.dequantize(q_data, Nc)
        # A_q = At[:, path_idx_q]

        H_q = recon_chan(A_q, g_q, L)
        Fopt_q, Wopt_q = getChannel(H_q, K, Nr, Ns)
        FRF_q, FBB_q = MO_AltMin(Fopt_q, NRF)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_q @ FBB_q[:, :, k], 'fro')
            if norm_f > 1e-9:
                FBB_q[:, :, k] = (np.sqrt(Ns) / norm_f) * FBB_q[:, :, k]

        rates_q_vs_B[b_idx, si] = calc_rate(H_user, FRF_q, FBB_q, Wopt_q, snr, K, Ns, Nc)

# --- Results & Plotting ---
plt.figure(figsize=(10, 7))

rate_p_avg = np.mean(rates_p)
rate_e_avg = np.mean(rates_e)
rate_q_avg = np.mean(rates_q_vs_B, axis=1)

print(f"\n--- Fixed SNR = {SNR_DB} dB ---")
print(f"Perfect CSI Rate: {rate_p_avg:.4f}")
print(f"Estimated CSI (No Quantization) Rate: {rate_e_avg:.4f}")
print("\n--- Limited Feedback Rate vs. B ---")
for b, rate in zip(B_vals, rate_q_avg):
    print(f"B = {b} bits, Rate: {rate:.4f}")

plt.axhline(y=rate_p_avg, color='b', linestyle='-', label='Perfect CSI (Upper Bound)')
plt.axhline(y=rate_e_avg, color='g', linestyle='--', label='Estimated CSI (No Quantization)')
plt.plot(B_vals, rate_q_avg, 'r-.s', label='Estimated CSI with Limited Feedback')

plt.xscale('log', base=2)
plt.grid(True, which='both')
plt.xlabel('Total Feedback Bits (B)')
plt.ylabel('Spectral Efficiency (bits/s/Hz)')
plt.title(f'Performance vs. Feedback Bits at SNR = {SNR_DB} dB')
plt.legend()
plt.xticks(B_vals, labels=B_vals)
plt.show()