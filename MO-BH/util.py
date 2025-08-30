import numpy as np


def get_Fopt_Wopt(H_channel, K, Nr, Ns):
    """
    通过对每个子载波的信道矩阵进行SVD，计算最优的全数字预编码器Fopt和组合器Wopt。
    """
    # H_channel 的维度是 (K*Nr, Nc, Nt)
    # _, Nc, Nt = H_channel.shape
    Nc = H_channel.shape[1]
    Nt = H_channel.shape[2]

    Fopt = np.zeros((Nt, Ns, Nc), dtype=np.complex128)
    Wopt = np.zeros((K * Nr, Ns, Nc), dtype=np.complex128)
    for k in range(Nc):
        # H_k 的维度是 (K*Nr, Nt)
        H_k = H_channel[:, k, :].reshape(K * Nr, Nt)
        U, _, Vh = np.linalg.svd(H_k, full_matrices=False)
        Fopt[:, :, k] = Vh.conj().T[:, :Ns]
        Wopt[:, :, k] = U[:, :Ns]
    return Fopt, Wopt


def calculate_sum_rate(H_true, FRF, FBB, W, snr_val, K, Ns, Nc):
    """
    计算给定混合预编码器和接收组合器下的系统和速率。

    Args:
        H_true (np.ndarray): 真实的信道矩阵, shape (K, Nc, Nt)。
        FRF (np.ndarray): 模拟预编码器, shape (Nt, NRF)。
        FBB (np.ndarray): 数字预编码器, shape (NRF, Ns, Nc)。
        W (np.ndarray): 接收组合器, shape (K*Nr, Ns, Nc)。
        snr_val (float): 线性信噪比。
        K (int): 用户数。
        Ns (int): 数据流数。
        Nc (int): 子载波数。

    Returns:
        float: 平均每个子载波的频谱效率 (bits/s/Hz)。
    """
    rate = 0.0
    Nr = H_true.shape[0] // K  # 每个用户的接收天线数

    for k_sub in range(Nc):
        # 获取当前子载波的预编码器、组合器和真实信道
        F_k = FRF @ FBB[:, :, k_sub]
        W_k = W[:, :, k_sub]
        H_k = H_true[:, k_sub, :].reshape(K * Nr, -1)  # 展平为 (K*Nr, Nt)

        # 计算有效信道 H_eff = W^H * H * F
        H_eff = W_k.conj().T @ H_k @ F_k

        # 容量公式变为 R = log2(det(I + (SNR/Ns) * H_eff * H_eff'))
        term_inside_det = np.eye(Ns, dtype=np.complex128) + (snr_val / Ns) * (H_eff @ H_eff.conj().T)

        # 使用 slogdet 计算 log(det(...)) 以保证数值稳定性
        sign, logdet = np.linalg.slogdet(term_inside_det)

        # 确保行列式为正
        if sign > 0:
            # np.log 是自然对数，需要除以 np.log(2) 转换为以2为底
            rate += logdet / np.log(2)

        # # 遍历每个数据流 (在我们的场景下，每个流对应一个用户)
        # for i in range(Ns):
        #     # 信号功率是有效信道对角线元素的模平方
        #     signal_power = np.abs(H_eff[i, i]) ** 2
        #
        #     # 干扰功率是该行非对角线元素的模平方和
        #     interference_power = np.sum(np.abs(H_eff[i, :]) ** 2) - signal_power
        #
        #     # 噪声功率。因为总发射功率归一化为Ns，所以噪声项要乘以Ns
        #     # W的列是单位正交的，所以 ||w_i||^2 = 1，噪声没有被放大
        #     noise_term = Ns / snr_val
        #
        #     sinr = signal_power / (interference_power + noise_term)
        #     rate += np.log2(1 + sinr)

    # 返回所有子载波的平均速率
    return np.real(rate) / Nc