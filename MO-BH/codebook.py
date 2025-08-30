import numpy as np


def generate_upa_codebook(Nt, n_angles_per_dim):
    """
    为均匀平面阵列 (UPA) 生成一个码本 (字典)。

    码本由对应于一系列离散方位角和俯仰角的阵列响应向量构成。
    此函数使用了参考论文中给出的标准公式来计算向量。

    Args:
        Nt (tuple): 一个元组，包含每个轴上的天线数量，例如 (8, 8)。
        n_angles_per_dim (int): 在每个角度维度（方位角和俯仰角）上采样的点数。
                               码本中的原子总数将是 n_angles_per_dim * n_angles_per_dim。

    Returns:
        np.ndarray: 码本矩阵 At，其维度为 (Nt_y * Nt_z, n_angles_per_dim * n_angles_per_dim)。
    """
    Nt_y, Nt_z = Nt
    N_t = Nt_y * Nt_z

    # 为 UPA 创建天线索引网格 (论文公式中的 p 和 q) 
    p_indices, q_indices = np.meshgrid(
        np.arange(Nt_y),
        np.arange(Nt_z)
    )

    # 将索引展平以便于进行向量化计算
    p_flat = p_indices.flatten()
    q_flat = q_indices.flatten()

    # 创建角度网格
    # 方位角 (phi) 的范围通常在 [-pi/2, pi/2]
    phi_values = np.linspace(-np.pi / 2, np.pi / 2, n_angles_per_dim, endpoint=True)
    # 俯仰角 (theta) 的范围通常在 [0, pi]
    theta_values = np.linspace(0, np.pi, n_angles_per_dim, endpoint=True)

    # 码本中的原子（列）总数
    total_atoms = n_angles_per_dim * n_angles_per_dim
    codebook = np.zeros((N_t, total_atoms), dtype=np.complex128)

    atom_idx = 0
    # 遍历所有角度组合
    for phi in phi_values:
        for theta in theta_values:
            # 这是论文中公式(4)的指数部分，假设天线间距 d = lambda / 2 
            # 对应的公式为: e^{j * pi * (p * sin(phi) * sin(theta) + q * cos(theta))}
            exponent = 1j * np.pi * (p_flat * np.sin(phi) * np.sin(theta) + q_flat * np.cos(theta))
            atom = np.exp(exponent)

            # 对每个原子进行归一化，使其具有单位范数
            normalized_atom = atom / np.sqrt(N_t)

            codebook[:, atom_idx] = normalized_atom
            atom_idx += 1

    return codebook
