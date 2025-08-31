import numpy as np


def getChannel(H, K, Nr, Ns):
    """
    通过对每个子载波的信道矩阵进行SVD，计算最优的全数字预编码器Fopt和组合器Wopt。
    """
    # H 的维度是 (K*Nr, Nc, Nt)
    _, Nc, Nt = H.shape
    Fopt = np.zeros((Nt, Ns, Nc), dtype=np.complex128)
    Wopt = np.zeros((K * Nr, Ns, Nc), dtype=np.complex128)
    for k in range(Nc):
        # H_k 的维度是 (K*Nr, Nt)
        H_k = H[:, k, :].reshape(K * Nr, Nt)
        U, _, Vh = np.linalg.svd(H_k, full_matrices=False)
        Fopt[:, :, k] = Vh.conj().T[:, :Ns]
        Wopt[:, :, k] = U[:, :Ns]
    return Fopt, Wopt


def calc_rate(H, FRF, FBB, W, snr, K, Ns, Nc):
    """
    计算给定混合预编码器和接收组合器下的系统和速率。
    Args:
        H (np.ndarray): 真实的信道矩阵, shape (K, Nc, Nt)。
        FRF (np.ndarray): 模拟预编码器, shape (Nt, NRF)。
        FBB (np.ndarray): 数字预编码器, shape (NRF, Ns, Nc)。
        W (np.ndarray): 接收组合器, shape (K*Nr, Ns, Nc)。
        snr (float): 线性信噪比。
        K (int): 用户数。
        Ns (int): 数据流数。
        Nc (int): 子载波数。
    Returns:
        float: 平均每个子载波的频谱效率 (bits/s/Hz)。
    """
    rate = 0.0
    Nr = H.shape[0] // K
    for k in range(Nc):
        Fk = FRF @ FBB[:, :, k]
        Wk = W[:, :, k]
        Hk = H[:, k, :].reshape(K * Nr, -1)
        H_eff = Wk.conj().T @ Hk @ Fk
        det_term = np.eye(Ns, dtype=np.complex128) + (snr / Ns) * (H_eff @ H_eff.conj().T)
        sign, logdet = np.linalg.slogdet(det_term)
        if sign > 0:
            rate += logdet / np.log(2)
        # 遍历每个数据流的计算方法
        # for i in range(Ns):
        #     signal_power = np.abs(H_eff[i, i]) ** 2
        #     interference_power = np.sum(np.abs(H_eff[i, :]) ** 2) - signal_power
        #     # 噪声功率。因为总发射功率归一化为Ns，所以噪声项要乘以Ns
        #     # W的列是单位正交的，所以 ||w_i||^2 = 1，噪声没有被放大
        #     noise_term = Ns / snr_val
        #     sinr = signal_power / (interference_power + noise_term)
        #     rate += np.log2(1 + sinr)
    return np.real(rate) / Nc

def gen_steer_vec(nt_dims, phi, theta):
    """
    根据给定的方位角(phi)和俯仰角(theta)，为UPA阵列生成一个方向向量。

    Args:
        nt_dims (tuple): (Nt_y, Nt_z) 每个轴上的天线数。
        phi (float): 方位角 (azimuth) in radians.
        theta (float): 俯仰角 (elevation) in radians.

    Returns:
        np.ndarray: 归一化后的方向向量, shape (Nt_y * Nt_z, 1)。
    """
    Nt_y, Nt_z = nt_dims
    N_t = Nt_y * Nt_z

    p_indices, q_indices = np.meshgrid(np.arange(Nt_y), np.arange(Nt_z))
    p_flat = p_indices.flatten()
    q_flat = q_indices.flatten()

    # 天线间距 d = lambda / 2 . 参考[1]公式(4)
    exponent = 1j * np.pi * (p_flat * np.sin(phi) * np.sin(theta) + q_flat * np.cos(theta))
    atom = np.exp(exponent)
    normalized_atom = atom / np.sqrt(N_t)
    return normalized_atom.reshape(-1, 1)


def idx_to_angles(idx, n_ang):
    """
    根据字典中的索引，反向计算出对应的phi和theta角。
    Args:
        idx (int): 字典中的原子索引。
        n_ang (int): 每个角度维度上的采样点数。
    Returns:
        tuple: (phi, theta) in radians.
    """
    phi_v = np.linspace(-np.pi / 2, np.pi / 2, n_ang)
    theta_v = np.linspace(0, np.pi, n_ang)
    phi_idx = idx // n_ang
    theta_idx = idx % n_ang
    return phi_v[phi_idx], theta_v[theta_idx]