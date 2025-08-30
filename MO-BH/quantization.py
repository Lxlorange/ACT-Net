# quantization.py

import numpy as np
import math

def lloyd_max(training_data, num_bits, iterations=20):
    """
    使用劳埃德-麦克斯算法为给定的训练数据设计一个标量量化器码本。
    (此函数无需修改)
    """
    if num_bits <= 0:
        return np.array([0.0])

    num_levels = 2 ** num_bits
    codebook = np.linspace(np.min(training_data), np.max(training_data), num_levels)
    for _ in range(iterations):
        boundaries = (codebook[:-1] + codebook[1:]) / 2.0
        partition_indices = np.digitize(training_data, boundaries)
        new_codebook = np.array([
            training_data[partition_indices == i].mean()
            for i in range(num_levels)
        ])
        nan_indices = np.isnan(new_codebook)
        new_codebook[nan_indices] = codebook[nan_indices]
        if np.allclose(codebook, new_codebook):
            break
        codebook = new_codebook
    return np.sort(codebook)


def quantize_values(values, codebook):
    """将浮点数值量化为码本中的索引。"""
    if len(codebook) <= 1:
        return np.zeros_like(values, dtype=int)
    diff = np.abs(values[..., np.newaxis] - codebook)
    indices = np.argmin(diff, axis=-1)
    return indices


def dequantize_values(indices, codebook):
    return codebook[indices]


class AngleGainQuantizer:
    """
    一个封装了物理角度和增益量化全过程的类。
    """

    def __init__(self, B_total, L, K, training_gains_avg, training_phis, training_thetas):
        self.B_total = B_total
        self.L = L
        self.K = K

        print("--- 量化器初始化 (策略: 物理角度+平均增益) ---")
        self._allocate_bits()

        # 为每一类参数训练独立的码本
        self.gain_codebook_real = lloyd_max(training_gains_avg.real.flatten(), self.b_gain)
        self.gain_codebook_imag = lloyd_max(training_gains_avg.imag.flatten(), self.b_gain)
        self.phi_codebook = lloyd_max(training_phis, self.b_phi)
        self.theta_codebook = lloyd_max(training_thetas, self.b_theta)
        print("所有参数的码本训练完毕。")
        print("------------------------------------------------")

    def _allocate_bits(self):
        """将总比特B均匀分配给所有需要反馈的物理参数。"""
        # 总共有 L*K个复数增益(2*L*K个实数) + L个phi角 + L个theta角
        self.num_params = (2 * self.L * self.K) + self.L + self.L

        if self.num_params > 0:
            bits_per_param = self.B_total // self.num_params
        else:
            bits_per_param = 0

        # 简单均匀分配
        self.b_gain = bits_per_param
        self.b_phi = bits_per_param
        self.b_theta = bits_per_param

        print("比特分配结果:")
        print(f"  - 总比特 B = {self.B_total}")
        print(f"  - 需量化参数总数 = {self.num_params}")
        print(f"  - 每个参数(增益实/虚部, phi, theta)分到: {bits_per_param} bits")

    def quantize(self, G_avg, phis, thetas):
        """UE端执行：量化所有参数并返回索引字典。"""
        quantized_data = {
            'gains_real': quantize_values(G_avg.real, self.gain_codebook_real),
            'gains_imag': quantize_values(G_avg.imag, self.gain_codebook_imag),
            'phis': quantize_values(phis, self.phi_codebook),
            'thetas': quantize_values(thetas, self.theta_codebook),
        }
        return quantized_data

    def dequantize(self, quantized_data):
        """BS端执行：根据索引字典恢复量化后的参数。"""
        g_real = dequantize_values(quantized_data['gains_real'], self.gain_codebook_real)
        g_imag = dequantize_values(quantized_data['gains_imag'], self.gain_codebook_imag)
        G_quant_avg = g_real + 1j * g_imag

        phis_quant = dequantize_values(quantized_data['phis'], self.phi_codebook)
        thetas_quant = dequantize_values(quantized_data['thetas'], self.theta_codebook)

        return G_quant_avg, phis_quant, thetas_quant