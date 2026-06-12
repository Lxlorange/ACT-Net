# sic.py

import numpy as np


def sic_hybrid_precoding_ofdm(H_sys, NRF, Ns, At):
    """
    基于连续干扰消除 (SIC) 的混合预编码算法 (全连接结构)。

    该算法首先基于所有子载波的平均信道信息，通过贪婪迭代的方式从码本At中
    选择出最佳的模拟波束构成FRF。然后，基于设计好的FRF，为每个子载波
    计算最优的数字预编码器FBB。

    Args:
        H_sys (np.ndarray):  系统信道矩阵，维度为 (K*Nr, Nc, Nt)。
                             (K:用户数, Nr:每用户天线数, Nc:子载波数, Nt:发射天线数)
        NRF (int):           射频链的数量。
        Ns (int):            数据流的数量。
        At (np.ndarray):     预定义的码本（字典），维度为 (Nt, N_atoms)。

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - FRF (np.ndarray): 模拟预编码矩阵 (频率无关)，维度为 (Nt, NRF)。
            - FBB (np.ndarray): 数字预编码矩阵 (频率选择性)，维度为 (NRF, Ns, Nc)。
    """
    # 获取维度信息
    _, Nc, Nt = H_sys.shape

    # --- 步骤 1: 设计频率无关的模拟预编码器 FRF ---
    G_eff = np.zeros((Nt, Nt), dtype=np.complex128)
    for k in range(Nc):
        H_k = H_sys[:, k, :]
        G_eff += H_k.conj().T @ H_k

    FRF = np.zeros((Nt, NRF), dtype=np.complex128)

    # 迭代 NRF 次，为每个 RF 链选择一个模拟波束
    for i in range(NRF):
        correlation = np.diagonal(At.conj().T @ G_eff @ At)
        best_idx = np.argmax(np.abs(correlation))

        a_i = At[:, best_idx].reshape(Nt, 1)
        FRF[:, i] = a_i.flatten()

        # --- 连续干扰消除 ---
        G_a = G_eff @ a_i
        num_term = G_a @ G_a.conj().T
        den_term = a_i.conj().T @ G_a

        if np.abs(den_term) > 1e-9:
            G_eff = G_eff - num_term / den_term[0, 0]

    # --- 步骤 2: 计算每个子载波的数字预编码器 FBB ---
    FBB = np.zeros((NRF, Ns, Nc), dtype=np.complex128)

    for k in range(Nc):
        H_k = H_sys[:, k, :]
        H_eff_k = H_k @ FRF
        _, _, Vh_eff = np.linalg.svd(H_eff_k, full_matrices=False)
        FBB[:, :, k] = Vh_eff.conj().T[:, :Ns]

    return FRF, FBB