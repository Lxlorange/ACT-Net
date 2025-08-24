# wmmse.py

import numpy as np
import config as cfg


def wmmse_precoder(H_list_users, noise_power):
    """
    使用WMMSE算法计算最大化和速率的全数字预编码器 F_BB[n]。
    """
    F_bb_list = [np.zeros((cfg.M, cfg.K), dtype=np.complex128) for _ in range(cfg.Nc)]

    for n in range(cfg.Nc):
        H_n = np.vstack([H_list_users[k][n, :, :] for k in range(cfg.K)])

        F_n = H_n.conj().T
        power_scale = np.sqrt(cfg.Pt / cfg.Nc) / np.linalg.norm(F_n, 'fro')
        F_n *= power_scale

        for _ in range(cfg.MAX_ITER_WMMSE):
            U = np.zeros(cfg.K, dtype=np.complex128)
            W = np.zeros(cfg.K, dtype=np.complex128)

            for k in range(cfg.K):
                h_k = H_n[k, :]

                # --- Bug修复：使用传入的正确噪声功率 ---
                interference_plus_noise = noise_power
                for j in range(cfg.K):
                    if j != k:
                        interference_plus_noise += np.abs(h_k @ F_n[:, j]) ** 2

                U[k] = (h_k @ F_n[:, k]) / interference_plus_noise
                W[k] = 1 / (1 - np.real(U[k] * (h_k @ F_n[:, k]).conj()))  # 修正为共轭

            B = np.zeros((cfg.M, cfg.M), dtype=np.complex128)
            # --- Bug修复：正则化项应与噪声功率相关 ---
            B += (cfg.K / (cfg.Pt / cfg.Nc)) * noise_power * np.eye(cfg.M)
            for k in range(cfg.K):
                h_k_herm = H_n[k, :].conj().T.reshape(cfg.M, 1)
                B += W[k] * np.abs(U[k]) ** 2 * (h_k_herm @ h_k_herm.T)

            # 求解更新后的 F
            inv_B = np.linalg.inv(B + 1e-9 * np.eye(cfg.M))
            for k in range(cfg.K):
                h_k_herm = H_n[k, :].conj().T
                F_n[:, k] = inv_B @ (W[k] * U[k].conj() * h_k_herm)

            power_scale = np.sqrt(cfg.Pt / cfg.Nc) / np.linalg.norm(F_n, 'fro')
            F_n *= power_scale

        F_bb_list[n] = F_n

    return F_bb_list