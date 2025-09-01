# swomp.py

import numpy as np
from util import idx_to_angles

def estimate_swomp(Y, X, At, L, n_angle):
    """
    使用SW-OMP算法估计稀疏毫米波信道的参数。
    该函数模拟UE端的信道估计过程。

    Args:
        Y (np.ndarray): UE接收到的、经过信道传输并带噪声的导频信号。
                        维度为 (K, Nc, Q)，Q是导频长度。
        X (np.ndarray): BS发送的、UE已知的导频矩阵。维度为 (Q, Nt)。
        At (np.ndarray): 字典矩阵 (码本)。维度为 (Nt, N_atoms)。
        L (int): 信道的稀疏度，即要估计的路径数量。

    Returns:
        tuple: (A_est, G_est)
            - A_est (np.ndarray): 估计出的路径对应的导向矢量矩阵。维度为 (Nt, L)。
            - G_est (np.ndarray): 估计出的每条路径在每个子载波上的复数增益。
                                维度为 (K, Nc, L)。
    """
    K, Nc, Q = Y.shape
    Nt, N_atoms = At.shape
    Phi = X @ At # Shape: (Q, Nt) @ (Nt, N_atoms) -> (Q, N_atoms)
    # OMP算法现在是求解 Y ≈ G @ Phi.T
    # 为了匹配维度，我们将问题看作求解 Y.T ≈ Phi @ G.T
    # 即 y_k_n ≈ Phi @ g_k_n, 其中 y_k_n 是 Qx1, g_k_n 是 N_atoms x 1 (稀疏)
    path_idx_list = []
    est_phi = []
    est_theta = []
    A_est = np.empty((Nt, 0), dtype=np.complex128)
    res = Y.copy() # Shape: (K, Nc, Q)

    for _ in range(L):
        # 匹配: 将残差投影到感知矩阵 Phi 上
        # residual shape (K, Nc, Q), Phi.conj() shape (Q, N_atoms)
        # P shape will be (K, Nc, N_atoms)
        P = np.tensordot(res, Phi.conj(), axes=([2], [0]))

        # 找到在所有用户和子载波上能量总和最大的原子
        obj = np.sum(np.abs(P) ** 2, axis=(0, 1))
        best_idx = np.argmax(obj)

        # 记录选中的路径索引并更新导向矢量矩阵 A_est
        path_idx_list.append(best_idx)
        phi, theta = idx_to_angles(best_idx, n_angle)
        est_phi.append(phi)
        est_theta.append(theta)

        sel_atom = At[:, best_idx].reshape(-1, 1)
        A_est = np.hstack([A_est, sel_atom])

        # 从原始接收信号 Y 中求解
        # Y ≈ (G @ A_est.T) @ X.T = G @ (X @ A_est).T
        # A_est_pinv = np.linalg.pinv(A_est)
        # G_est_current = np.tensordot(Y, np.linalg.pinv(X.T).conj(), axes=([2],[1]))
        # G_est_current = np.tensordot(G_est_current, A_est_pinv.conj(), axes=([2],[1]))

        # 一个更稳健的LS求解方法
        Phi_L = X @ A_est
        Phi_L_pinv = np.linalg.pinv(Phi_L)
        # G_est_current shape (K, Nc, L)
        g_curr = np.tensordot(Y, Phi_L_pinv.conj().T, axes=([2], [0]))

        Y_recon = np.tensordot(g_curr, Phi_L.T, axes=([2], [0]))
        res = Y - Y_recon

    Phi_L = X @ A_est
    g_est = np.tensordot(Y, np.linalg.pinv(Phi_L).conj().T, axes=([2], [0]))
    g_est = g_est.transpose(0, 2, 1)

    path_idx = np.array(path_idx_list, dtype=int)

    return g_est, np.array(est_phi), np.array(est_theta), path_idx

def recon_chan(A_est, G_est, L):
    """
    使用估计出的导向矢量和增益重构信道矩阵。
    该函数模拟BS端的信道重构过程。

    Args:
        A_est (np.ndarray): 估计的导向矢量矩阵。维度 (Nt, L)。
        G_est (np.ndarray): 估计的复数增益矩阵。维度 (K, L, Nc)。

    Returns:
        np.ndarray: 重构出的信道矩阵 H_hat。维度 (K, Nc, Nt)。
    """
    # 使用爱因斯坦求和约定高效地进行矩阵乘法
    # K:用户, L:路径, N:子载波, T:天线
    # G_est (K,L,N) @ A_est (T,L) -> H_hat (K,N,T)
    H_hat = np.einsum('kln,tl->knt', G_est, A_est)
    if L > 0:
        H_hat = H_hat / np.sqrt(L)

    return H_hat