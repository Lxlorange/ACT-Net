# wmmse.py

import numpy as np
import config as cfg


def wmmse_precoder(H_list_users, noise_power):
    """
    Final corrected WMMSE algorithm using efficient scalar operations for the MISO case.
    """
    F_bb_list = [np.zeros((cfg.M, cfg.K), dtype=np.complex128) for _ in range(cfg.Nc)]

    for n in range(cfg.Nc):
        H_n = np.vstack([H_list_users[k][n, 0, :] for k in range(cfg.K)])

        # Initialize with MRT
        F_n = H_n.conj().T
        power_scale = np.sqrt(cfg.Pt / cfg.Nc) / np.linalg.norm(F_n, 'fro')
        F_n *= power_scale

        for _ in range(cfg.MAX_ITER_WMMSE):
            # --- This entire block is now replaced with efficient scalar calculations ---
            U = np.zeros(cfg.K, dtype=np.complex128)
            W = np.zeros(cfg.K, dtype=np.complex128)

            # 1. For each user, calculate its scalar MMSE equalizer U[k] and weight W[k]
            for k in range(cfg.K):
                h_k = H_n[k, :]

                # Desired signal channel gain
                signal_gain = h_k @ F_n[:, k]

                # Total received power at user k (signal + interference + noise)
                total_rx_power = noise_power
                for j in range(cfg.K):
                    total_rx_power += np.abs(h_k @ F_n[:, j]) ** 2

                # MMSE Equalizer (scalar)
                U[k] = signal_gain.conj() / total_rx_power

                # MMSE (scalar)
                mse_k = 1 - np.real(U[k] * signal_gain)

                # WMMSE Weight (scalar)
                W[k] = 1 / mse_k

            # 2. Update the precoder F_n
            # First term for the matrix to be inverted
            B = np.zeros((cfg.M, cfg.M), dtype=np.complex128)
            for j in range(cfg.K):
                h_j_herm = H_n[j, :].conj().T.reshape(cfg.M, 1)
                B += W[j] * np.abs(U[j]) ** 2 * (h_j_herm @ h_j_herm.T)

            # Second term for the matrix to be inverted (regularization)
            # This term comes from the derivative of the power constraint
            mu_I = np.zeros((cfg.M, cfg.M), dtype=np.complex128)
            for j in range(cfg.K):
                mu_I += W[j] * np.abs(U[j]) ** 2 * np.eye(cfg.M)

            # Invert the combined matrix
            inv_term = np.linalg.inv(B + mu_I)

            # Update each user's precoding vector
            for k in range(cfg.K):
                h_k_herm = H_n[k, :].conj().T
                F_n[:, k] = inv_term @ (W[k] * U[k].conj() * h_k_herm)

            # 3. Power normalization
            power_scale = np.sqrt(cfg.Pt / cfg.Nc) / np.linalg.norm(F_n, 'fro')
            F_n *= power_scale

        F_bb_list[n] = F_n

    return F_bb_list