# quantization.py

import numpy as np
import math

def lloyd_max(x, b, iters=20):
    """使用劳埃德-麦克斯算法为给定的训练数据设计一个标量量化器码本。"""
    if b <= 0:
        return np.array([0.0])

    n_levels = 2 ** b
    cb = np.linspace(np.min(x), np.max(x), n_levels)
    for _ in range(iters):
        bnd = (cb[:-1] + cb[1:]) / 2.0
        idx = np.digitize(x, bnd)
        new_cb = np.array([x[idx == i].mean() for i in range(n_levels)])
        nan_idx = np.isnan(new_cb)
        new_cb[nan_idx] = cb[nan_idx]
        if np.allclose(cb, new_cb):
            break
        cb = new_cb
    return np.sort(cb)


def quantize(vals, cb):
    """将浮点数值量化为码本中的索引。"""
    if len(cb) <= 1:
        return np.zeros_like(vals, dtype=int)
    diff = np.abs(vals[..., np.newaxis] - cb)
    idx = np.argmin(diff, axis=-1)
    return idx


def dequantize(idx, cb):
    return cb[idx]


class AGQuantizer:
    """
    一个封装了物理角度和增益量化全过程的类。
    """
    def __init__(self, B, L, K, train_g, train_phi, train_theta):
        self.B = B
        self.L = L
        self.K = K

        self._alloc_bits()

        self.g_cb_r = lloyd_max(train_g.real.flatten(), self.b_g)
        self.g_cb_i = lloyd_max(train_g.imag.flatten(), self.b_g)
        self.phi_cb = lloyd_max(train_phi, self.b_phi)
        self.theta_cb = lloyd_max(train_theta, self.b_theta)

    def _alloc_bits(self):
        """Uniformly allocates total bits B to all parameters."""
        n_params = (2 * self.L * self.K) + self.L + self.L
        b_param = self.B // n_params if n_params > 0 else 0
        self.b_g = b_param
        self.b_phi = b_param
        self.b_theta = b_param

    def quantize(self, g_avg, phis, thetas):
        """Quantizes all parameters and returns a dictionary of indices."""
        q_data = {
            'g_r': quantize(g_avg.real, self.g_cb_r),
            'g_i': quantize(g_avg.imag, self.g_cb_i),
            'phis': quantize(phis, self.phi_cb),
            'thetas': quantize(thetas, self.theta_cb),
        }
        return q_data

    def dequantize(self, q_data):
        """Recovers quantized parameters from a dictionary of indices."""
        g_r = dequantize(q_data['g_r'], self.g_cb_r)
        g_i = dequantize(q_data['g_i'], self.g_cb_i)
        g_q_avg = g_r + 1j * g_i
        phi_q = dequantize(q_data['phis'], self.phi_cb)
        theta_q = dequantize(q_data['thetas'], self.theta_cb)
        return g_q_avg, phi_q, theta_q


class IdxGainQuantizer:
    """
    一个封装了信道参数量化和反馈全过程的类。
    策略：量化路径索引 (Index) 和 平均信道增益 (Gain)。
    """

    def __init__(self, B, L, K, n_atoms, train_g):
        self.B = B
        self.L = L
        self.K = K
        self.n_atoms = n_atoms

        self._alloc_bits()

        if self.b_g > 0:
            g_r = train_g.real.flatten()
            g_i = train_g.imag.flatten()
            self.g_cb_r = lloyd_max(g_r, self.b_g)
            self.g_cb_i = lloyd_max(g_i, self.b_g)
        else:
            self.g_cb_r = np.array([0.0])
            self.g_cb_i = np.array([0.0])

    def _alloc_bits(self):
        """为路径索引和平均增益分配比特。"""
        self.b_idx = math.ceil(math.log2(self.n_atoms))
        B_idx = self.L * self.b_idx
        B_g = self.B - B_idx
        if B_g < 0:
            B_g = 0

        n_g_vals = 2 * self.L * self.K  # 实部 + 虚部
        self.b_g = B_g // n_g_vals if n_g_vals > 0 else 0

    def quantize(self, path_idx, g_est):
        """UE端执行：量化路径索引和计算出的平均增益。"""
        g_avg = np.mean(g_est, axis=2)
        q_data = {
            'path_idx': path_idx,
            'g_r_idx': quantize(g_avg.real, self.g_cb_r),
            'g_i_idx': quantize(g_avg.imag, self.g_cb_i)
        }
        return q_data

    def dequantize(self, q_data, Nc):
        """BS端执行：恢复索引和增益，并将平均增益扩展至所有子载波。"""
        path_idx = q_data['path_idx']

        g_r = dequantize(q_data['g_r_idx'], self.g_cb_r)
        g_i = dequantize(q_data['g_i_idx'], self.g_cb_i)
        g_q_avg = g_r + 1j * g_i

        # 将平均增益扩展(广播)至所有Nc个子载波
        g_q = np.tile(np.expand_dims(g_q_avg, axis=2), (1, 1, Nc))

        return path_idx, g_q