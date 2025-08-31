import numpy as np


def gen_upa_cb(nt_dims, n_ang):
    """
    为均匀平面阵列 (UPA) 生成一个码本 (字典)。
    码本由对应于一系列离散方位角和俯仰角的阵列响应向量构成。
    此函数使用了参考论文中给出的标准公式来计算向量。
    Args:
        nt_dims (tuple): 一个元组，包含每个轴上的天线数量，例如 (8, 8)。
        n_ang (int): 在每个角度维度（方位角和俯仰角）上采样的点数。
                               码本中的原子总数将是 n_angles_per_dim * n_angles_per_dim。
    Returns:
        np.ndarray: 码本矩阵 At，其维度为 (Nt_y * Nt_z, n_angles_per_dim * n_angles_per_dim)。
    """
    Nt_y, Nt_z = nt_dims
    Nt = Nt_y * Nt_z
    p_idx, q_idx = np.meshgrid(np.arange(Nt_y), np.arange(Nt_z))
    p_flat = p_idx.flatten()
    q_flat = q_idx.flatten()
    phi_v = np.linspace(-np.pi / 2, np.pi / 2, n_ang, endpoint=True)
    theta_v = np.linspace(0, np.pi, n_ang, endpoint=True)
    n_atoms = n_ang * n_ang
    cb = np.zeros((Nt, n_atoms), dtype=np.complex128)
    atom_idx = 0
    for phi in phi_v:
        for theta in theta_v:
            expnt = 1j * np.pi * (p_flat * np.sin(phi) * np.sin(theta) + q_flat * np.cos(theta))
            atom = np.exp(expnt)
            norm_atom = atom / np.sqrt(Nt)
            cb[:, atom_idx] = norm_atom
            atom_idx += 1
    return cb
