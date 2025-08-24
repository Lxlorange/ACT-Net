import numpy as np
import config as cfg
import pymanopt
from pymanopt.manifolds import ComplexCircle
from pymanopt.optimizers import ConjugateGradient


class HybridPrecoder_MU_MO:
    def __init__(self):
        self.F_RF = None
        self.F_BB_list = [None] * cfg.Nc

    def _create_problem_components(self, F_BB_list, F_opt_list):
        manifold = ComplexCircle(cfg.M * cfg.K)

        @pymanopt.function.numpy(manifold)
        def cost(f_rf_vec):
            F_RF = f_rf_vec.reshape(cfg.M, cfg.K)
            total_error = sum(np.linalg.norm(F_opt_list[n] - F_RF @ F_BB_list[n], 'fro') ** 2 for n in range(cfg.Nc))
            return total_error

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(f_rf_vec):
            F_RF = f_rf_vec.reshape(cfg.M, cfg.K)
            grad_matrix = np.zeros_like(F_RF, dtype=np.complex128)
            for n in range(cfg.Nc):
                F_BB_n = F_BB_list[n]
                F_opt_n = F_opt_list[n]
                grad_matrix += -2 * (F_opt_n @ F_BB_n.conj().T - F_RF @ F_BB_n @ F_BB_n.conj().T)
            return grad_matrix.flatten()

        return manifold, cost, euclidean_gradient

    def _update_F_RF_manifold(self, F_RF, F_BB_list, F_opt_list):
        manifold, cost_func, grad_func = self._create_problem_components(F_BB_list, F_opt_list)
        problem = pymanopt.Problem(manifold=manifold, cost=cost_func, euclidean_gradient=grad_func)
        solver = ConjugateGradient(max_iterations=cfg.MAX_ITER_MANIFOLD, log_verbosity=0)
        f_rf_initial_vec = F_RF.flatten()
        result = solver.run(problem, initial_point=f_rf_initial_vec)
        f_rf_opt_vec = result.point
        return f_rf_opt_vec.reshape(cfg.M, cfg.K)

    def compute(self, F_opt_list):
        # --- 核心修正：使用PCA为 F_RF 提供高质量的初始值 ---

        # 1. 将所有子载波的 F_opt 拼接成一个大矩阵
        F_opt_stacked = np.hstack(F_opt_list)  # Shape: (M, K * Nc) -> (64, 64)

        # 2. 对这个大矩阵进行SVD，找到其主要的左奇异向量
        # U 的列向量是 F_opt_stacked 列空间的一组正交基
        # 前 K 个左奇异向量是能量最集中的方向
        U, _, _ = np.linalg.svd(F_opt_stacked, full_matrices=False)

        # 3. 用前 K 个左奇异向量作为 F_RF 的初始方向
        F_RF_unquantized = U[:, :cfg.K]

        # 4. 提取相位，满足单位模约束，作为最终的初始 F_RF
        initial_phases = np.angle(F_RF_unquantized)
        self.F_RF = np.exp(1j * initial_phases)

        # --- 随机初始化（旧方法，注释掉）---
        # random_phases = np.random.uniform(0, 2 * np.pi, (cfg.M, cfg.K))
        # self.F_RF = np.exp(1j * random_phases)

        # --- 交替最小化循环保持不变 ---
        for i in range(cfg.MAX_ITER_ALTMIN):
            for n in range(cfg.Nc):
                self.F_BB_list[n] = np.linalg.pinv(self.F_RF) @ F_opt_list[n]

            self.F_RF = self._update_F_RF_manifold(self.F_RF, self.F_BB_list, F_opt_list)

        # --- 功率归一化保持不变 ---
        final_F_BB_list = [np.zeros_like(F_BB_n, dtype=np.complex128) for F_BB_n in self.F_BB_list]
        for n in range(cfg.Nc):
            F_BB_n = self.F_BB_list[n]
            # 这里的归一化需要针对整个预编码矩阵 F_H = F_RF @ F_BB_n
            norm_val = np.linalg.norm(self.F_RF @ F_BB_n, 'fro')
            if norm_val > 1e-9:
                # 缩放数字部分以满足总功率约束
                power_scaling = np.sqrt(cfg.Pt / cfg.Nc) / norm_val
                final_F_BB_list[n] = F_BB_n * power_scaling
            else:
                final_F_BB_list[n] = F_BB_n

        return self.F_RF, final_F_BB_list


def calculate_sum_rate(H_list, F_RF, F_BB_list, noise_power):
    total_rate = 0
    for n in range(cfg.Nc):
        H_n = H_list[n]
        F_BB_n = F_BB_list[n]
        F_n = F_RF @ F_BB_n
        for k in range(cfg.K):
            signal_vector = H_n[:, k].conj().T @ F_n[:, k]
            signal_power = np.abs(signal_vector) ** 2
            interference_power = 0
            for j in range(cfg.K):
                if j != k:
                    interference_vector = H_n[:, k].conj().T @ F_n[:, j]
                    interference_power += np.abs(interference_vector) ** 2
            sinr = signal_power / (interference_power + noise_power)
            total_rate += np.log2(1 + sinr)
    return total_rate / cfg.Nc