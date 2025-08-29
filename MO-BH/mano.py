import numpy as np
import pymanopt
from pymanopt.manifolds import ComplexCircle, Product
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient

def sig_manif(Fopt, FRF, FBB):
    """
    该函数使用流形优化来求解固定FBB时的最优FRF。
    它采用了预计算的方式来加速代价函数和梯度的计算。

    Args:
        Fopt (np.ndarray): 理想全数字预编码器 (Nt, Ns, K)。
        FRF (np.ndarray): 当前的模拟预编码器，作为优化的初始点 (Nt, NRF)。
        FBB (np.ndarray): 当前的数字预编码器 (NRF, Ns, K)。

    Returns:
        tuple: (FRF_new, cost_new)
            - FRF_new (np.ndarray): 优化后的新模拟预编码器。
            - cost_new (float): 优化后的目标函数值。
    """
    # Nt, NRF = FRF.shape
    if not isinstance(FRF, list) or len(FRF) == 0:
        raise ValueError("输入必须是一个非空列表")

    NRF = len(FRF)
    Nt = FRF[0].shape[0]
    _, _, K = FBB.shape

    C1_list = []
    C2_list = []
    C3_list = []
    C4_list = []

    for k in range(K):
        temp = Fopt[:, :, k]
        temp_vec = temp.flatten(order='F')  # 按列向量化

        A = np.kron(FBB[:, :, k].transpose(), np.eye(Nt))

        # 逐个计算并存储
        C1_list.append(temp_vec.conj().T @ A)
        C2_list.append(A.conj().T @ temp_vec)
        C3_list.append(A.conj().T @ A)
        C4_list.append(np.linalg.norm(temp, 'fro') ** 2)

    # 累加得到最终的常数项 B1, B2, B3, B4
    B1 = np.sum(np.array(C1_list), axis=0)
    B2 = np.sum(np.array(C2_list), axis=0)
    B3 = np.sum(np.array(C3_list), axis=0)
    B4 = np.sum(np.array(C4_list))

    # 定义流形， Pymanopt 中直接使用矩阵维度
    manifold = Product([ComplexCircle(Nt) for _ in range(NRF)])

    # 定义代价函数和欧几里得梯度
    # @pymanopt.function.numpy 是一个装饰器，用于告知pymanopt这是一个基于numpy的函数
    @pymanopt.function.numpy(manifold)
    def cost(x1, x2):
        X_list = [x1, x2]
        X_matrix = np.hstack(X_list)
        x_vec = X_matrix.flatten(order='F')
        cost_val = -B1 @ x_vec - x_vec.conj().T @ B2 + np.trace(
            B3 @ x_vec.reshape(-1, 1) @ x_vec.conj().T.reshape(1, -1)) + B4
        # cost_val = np.real(x_vec.conj().T @ B3 @ x_vec - 2 * (B1 @ x_vec)) + B4

        return np.real(cost_val)

    @pymanopt.function.numpy(manifold)
    def egrad(x1,x2):
        # 从列向量列表中重组出FRF矩阵
        X_list = [x1,x2]
        X_matrix = np.hstack(X_list)
        x_vec = X_matrix.flatten(order='F')
        grad_vec = -2 * B2 + 2 * (B3 @ x_vec)

        # 将梯度向量转换回矩阵形式
        grad_matrix = grad_vec.reshape(Nt, NRF, order='F')

        # 将梯度矩阵分解为列向量列表，以匹配Product流形的格式
        grad_list = [grad_matrix[:, i].flatten() for i in range(NRF)]
        return grad_list

    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=egrad)
    solver = ConjugateGradient(max_iterations=20, log_verbosity=0)
    result = solver.run(problem, initial_point=FRF)

    FRF_new_list = result.point
    FRF_new = np.column_stack(FRF_new_list)
    cost_new = result.cost
    return FRF_new, cost_new

def MO_AltMin(Fopt, NRF):
    """
    该函数通过交替最小化来计算混合预编码器。

    Args:
        Fopt (np.ndarray): 理想全数字预编码器，shape (Nt, Ns, K)。
        NRF (int): 射频链的数量。

    Returns:
        tuple: (FRF, FBB)
            - FRF (np.ndarray): 模拟预编码器, shape (Nt, NRF)。
            - FBB (np.ndarray): 数字预编码器, shape (NRF, Ns, K)。
    """

    Nt, Ns, K = Fopt.shape
    y = None
    FRF = np.exp(1j * np.random.uniform(0, 2 * np.pi, (Nt, NRF)))
    FBB = np.zeros((NRF, Ns, K), dtype=np.complex128)
    while y is None or abs(y[0] - y[1]) > 1e-1:

        y = [0.0, 0.0]
        FRF_pinv = np.linalg.pinv(FRF)
        for k in range(K):
            FBB[:, :, k] = FRF_pinv @ Fopt[:, :, k]

            error_k = np.linalg.norm(Fopt[:, :, k] - FRF @ FBB[:, :, k], 'fro') ** 2
            y[0] += error_k

        FRF_list = [FRF[:, i].flatten() for i in range(NRF)]
        FRF, y[1] = sig_manif(Fopt, FRF_list, FBB)

    return FRF, FBB

