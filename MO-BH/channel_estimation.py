# channel_estimation.py

import numpy as np
import config as cfg


class OMPChannelEstimator:
    def __init__(self):
        self.dictionary = self._generate_upa_dictionary()

    def _get_array_response(self, theta, phi):
        n1 = np.arange(cfg.Ny)
        n2 = np.arange(cfg.Nz)
        at1 = np.exp(-2j * np.pi * cfg.d_lambda_ratio * n1 * np.cos(theta) * np.sin(phi))
        at2 = np.exp(-2j * np.pi * cfg.d_lambda_ratio * n2 * np.sin(theta))
        return np.kron(at1, at2)

    def _generate_upa_dictionary(self, resolution=128):
        dictionary = np.zeros((cfg.M, resolution * resolution), dtype=np.complex128)
        thetas = np.linspace(0, np.pi, resolution)
        phis = np.linspace(0, 2 * np.pi, resolution)
        idx = 0
        for theta in thetas:
            for phi in phis:
                dictionary[:, idx] = self._get_array_response(theta, phi)
                idx += 1
        return dictionary

    def estimate_channel_parameters(self, H_list_users_true, noise_power):
        # 1. 生成FDD下行导频 (来自论文模型)
        # Pilot matrix ~X has shape (M, Q)
        pilot_phases = np.random.uniform(0, 2 * np.pi, (cfg.M, cfg.Q))
        pilot_matrix = np.sqrt(cfg.Pt / (cfg.M * cfg.Nc)) * np.exp(1j * pilot_phases)

        estimated_params = []
        for k in range(cfg.K):
            # H_k_true is (Nc, 1, M) -> (Nc, M)
            H_k_true = H_list_users_true[k][:, 0, :]

            # 2. 模拟导频接收
            # Y_k is (Nc, Q)
            noise = np.sqrt(noise_power / 2) * (np.random.randn(cfg.Nc, cfg.Q) + 1j * np.random.randn(cfg.Nc, cfg.Q))
            Y_k = H_k_true @ pilot_matrix + noise

            # 3. OMP恢复 - 在频域平均的信道上估计角度
            y_k_avg = np.mean(Y_k, axis=0)  # (Q,)

            # 有效感知矩阵 A = pilot_matrix.T @ Dictionary
            A = pilot_matrix.T @ self.dictionary

            residual = y_k_avg
            support_indices = []

            for _ in range(cfg.Lp):
                correlation = np.abs(A.conj().T @ residual)
                new_idx = np.argmax(correlation)
                support_indices.append(new_idx)

                A_s = A[:, support_indices]
                gains_est = np.linalg.pinv(A_s) @ y_k_avg
                residual = y_k_avg - A_s @ gains_est

            # 4. 估计每个子载波的增益
            gains_per_subcarrier = []
            A_s_recon = self.dictionary[:, support_indices]
            for n in range(cfg.Nc):
                y_kn = Y_k[n, :]
                effective_sensing_matrix = pilot_matrix.T @ A_s_recon
                gains_n = np.linalg.pinv(effective_sensing_matrix) @ y_kn.T
                gains_per_subcarrier.append(gains_n)

            estimated_params.append({
                'angle_indices': np.array(support_indices),
                'gains_all_carriers': np.array(gains_per_subcarrier)
            })

        return estimated_params

    def reconstruct_channel_from_params(self, params):
        H_recon_list_users = [np.zeros((cfg.Nc, cfg.Nr, cfg.M), dtype=np.complex128) for _ in range(cfg.K)]

        for k in range(cfg.K):  # 遍历用户
            user_params = params[k]
            angle_indices = user_params['angle_indices']
            gains_all_carriers = user_params['gains_all_carriers']

            A_s = self.dictionary[:, angle_indices]

            for n in range(cfg.Nc):  # 遍历子载波
                gains_n = gains_all_carriers[n]
                # h_kn is (M,)
                h_kn = A_s @ gains_n
                H_recon_list_users[k][n, 0, :] = h_kn

        return H_recon_list_users