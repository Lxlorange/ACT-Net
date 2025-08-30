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
    """将浮点数值量化为码本中的索引。(此函数无需修改)"""
    if len(codebook) <= 1:
        return np.zeros_like(values, dtype=int)
    diff = np.abs(values[..., np.newaxis] - codebook)
    indices = np.argmin(diff, axis=-1)
    return indices


def dequantize_values(indices, codebook):
    """根据索引从码本中恢复量化值。(此函数无需修改)"""
    return codebook[indices]


class FeedbackQuantizer:
    """
    一个封装了信道参数量化和反馈全过程的类。
    (已更新为平均增益量化策略)
    """

    def __init__(self, B_total, L, K, N_atoms, training_gains_avg):
        self.B_total = B_total
        self.L = L
        self.K = K
        self.N_atoms = N_atoms

        print("--- 量化器初始化 (新策略: 平均增益) ---")
        self._allocate_bits()

        if self.b_gain > 0:
            print(f"为平均增益实部/虚部训练 {self.b_gain}-bit Lloyd-Max 码本...")
            g_real_avg = training_gains_avg.real.flatten()
            g_imag_avg = training_gains_avg.imag.flatten()
            self.gain_codebook_real = lloyd_max(g_real_avg, self.b_gain)
            self.gain_codebook_imag = lloyd_max(g_imag_avg, self.b_gain)
            print("增益码本训练完毕。")
        else:
            self.gain_codebook_real = np.array([0.0])
            self.gain_codebook_imag = np.array([0.0])
            print("警告: 没有足够比特分配给信道增益。")
        print("---------------------------------------")

    def _allocate_bits(self):
        """基于平均增益的均匀比特分配策略。"""
        # a. 为L个路径索引分配比特 (不变)
        self.b_index = math.ceil(math.log2(self.N_atoms))
        B_for_indices = self.L * self.b_index

        # b. 剩余比特分配给平均增益 (实部+虚部)
        B_for_gains = self.B_total - B_for_indices
        if B_for_gains < 0:
            B_for_gains = 0
            print(f"警告: 总比特数 B={self.B_total} 不足以反馈所有路径索引！")

        # NEW: 只需量化 K*L 个复数平均增益
        num_gain_values = 2 * self.L * self.K  # 实部 + 虚部

        if num_gain_values > 0:
            self.b_gain = B_for_gains // num_gain_values
        else:
            self.b_gain = 0

        print("比特分配结果:")
        print(f"  - 总比特 B = {self.B_total}")
        print(f"  - 每个路径索引: {self.b_index} bits (共 {B_for_indices} bits)")
        print(f"  - 需量化的平均增益值总数: {num_gain_values // 2} (复数)")
        print(f"  - 每个平均增益值(实/虚): {self.b_gain} bits")

    def quantize(self, path_indices, G_est):
        """UE端执行：计算平均增益，然后量化。"""
        # NEW: 先计算平均增益
        G_avg = np.mean(G_est, axis=2)  # 沿子载波维度(axis=2)求平均

        quantized_indices = path_indices
        quantized_gains_real_idx = quantize_values(G_avg.real, self.gain_codebook_real)
        quantized_gains_imag_idx = quantize_values(G_avg.imag, self.gain_codebook_imag)

        return quantized_indices, quantized_gains_real_idx, quantized_gains_imag_idx

    def dequantize(self, quantized_indices, quantized_gains_real_idx, quantized_gains_imag_idx, Nc):
        """BS端执行：恢复平均增益，并扩展至所有子载波。"""
        dequantized_indices = quantized_indices

        dequantized_gains_real = dequantize_values(quantized_gains_real_idx, self.gain_codebook_real)
        dequantized_gains_imag = dequantize_values(quantized_gains_imag_idx, self.gain_codebook_imag)

        G_quant_avg = dequantized_gains_real + 1j * dequantized_gains_imag

        # NEW: 将平均增益扩展(广播)至所有Nc个子载波
        # G_quant_avg shape: (K, L) -> G_quant shape: (K, L, Nc)
        G_quant = np.expand_dims(G_quant_avg, axis=2)
        G_quant = np.tile(G_quant, (1, 1, Nc))

        return dequantized_indices, G_quant