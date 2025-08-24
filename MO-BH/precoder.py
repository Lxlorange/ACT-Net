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
        """
        为pymanopt定义流形、代价函数和梯度的完整组件。
        """
        manifold = ComplexCircle(cfg.M * cfg.K)

        # 1. 定义代价函数 (Cost Function)
        @pymanopt.function.numpy(manifold)
        def cost(f_rf_vec):
            F_RF = f_rf_vec.reshape(cfg.M, cfg.K)
            total_error = sum(np.linalg.norm(F_opt_list[n] - F_RF @ F_BB_list[n], 'fro') ** 2 for n in range(cfg.Nc))
            return total_error

        # 2. 定义欧氏梯度函数 (Euclidean Gradient) - 这是本次的核心修正
        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(f_rf_vec):
            F_RF = f_rf_vec.reshape(cfg.M, cfg.K)

            # 使用 MO 论文公式 (24) 计算梯度
            grad_matrix = np.zeros_like(F_RF, dtype=np.complex128)
            for n in range(cfg.Nc):
                F_BB_n = F_BB_list[n]
                F_opt_n = F_opt_list[n]
                grad_matrix += -2 * (F_opt_n @ F_BB_n.conj().T - F_RF @ F_BB_n @ F_BB_n.conj().T)

            # pymanopt 需要返回一个和输入形状相同的向量
            return grad_matrix.flatten()

        return manifold, cost, euclidean_gradient

    def _update_F_RF_manifold(self, F_RF, F_BB_list, F_opt_list):
        """
        使用pymanopt的共轭梯度求解器来更新 F_RF。
        """
        manifold, cost_func, grad_func = self._create_problem_components(F_BB_list, F_opt_list)

        # 3. 定义问题时，同时传入代价和梯度
        problem = pymanopt.Problem(manifold=manifold, cost=cost_func, euclidean_gradient=grad_func)

        solver = ConjugateGradient(max_iterations=cfg.MAX_ITER_MANIFOLD, log_verbosity=0)
        f_rf_initial_vec = F_RF.flatten()

        # 运行求解器
        result = solver.run(problem, initial_point=f_rf_initial_vec)
        f_rf_opt_vec = result.point

        return f_rf_opt_vec.reshape(cfg.M, cfg.K)

    def compute(self, F_opt_list):
        # 这个函数保持不变
        # print("\nStarting MU MO-AltMin computation...")
        random_phases = np.random.uniform(0, 2 * np.pi, (cfg.M, cfg.K))
        self.F_RF = np.exp(1j * random_phases)

        for i in range(cfg.MAX_ITER_ALTMIN):
            for n in range(cfg.Nc):
                self.F_BB_list[n] = np.linalg.pinv(self.F_RF) @ F_opt_list[n]

            self.F_RF = self._update_F_RF_manifold(self.F_RF, self.F_BB_list, F_opt_list)

        for n in range(cfg.Nc):
            F_BB_n = self.F_BB_list[n]
            norm_val = np.linalg.norm(self.F_RF @ F_BB_n, 'fro')
            if norm_val > 1e-9:
                power_scaling = np.sqrt(cfg.Pt / cfg.Nc) / norm_val
                self.F_BB_list[n] = F_BB_n * power_scaling

        return self.F_RF, self.F_BB_list


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