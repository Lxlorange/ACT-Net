# _*_ coding : utf-8 _*_
# @Time :  15:09
# @Author : Lxl
# @File ： quantization
# @ProjectName : OFDM
# quantization.py

import numpy as np
import config as cfg


class UniformQuantizer:
    def __init__(self, num_bits, min_val, max_val):
        self.num_bits = num_bits
        self.min_val = min_val
        self.max_val = max_val
        self.num_levels = 2 ** num_bits
        self.step_size = (max_val - min_val) / self.num_levels
        # 构建码本
        self.codebook = np.linspace(min_val + self.step_size / 2, max_val - self.step_size / 2, self.num_levels)

    def quantize(self, value):
        """将值量化为码本中的索引"""
        value = np.clip(value, self.min_val, self.max_val)
        index = np.floor((value - self.min_val) / self.step_size)
        return int(np.clip(index, 0, self.num_levels - 1))

    def dequantize(self, index):
        """根据索引从码本中恢复值"""
        return self.codebook[index]


def quantize_channel_params(params):
    """对一组信道参数进行量化"""
    # 简单的范围假设，实际应用中应基于统计数据
    gain_quantizer = UniformQuantizer(cfg.FEEDBACK_BITS_PER_PARAM, -2, 2)
    # 角度索引已经是离散的，不需要量化，直接反馈

    quantized_params = []
    for user_p in params:
        q_gains_all_carriers = np.zeros_like(user_p['gains_all_carriers'], dtype=np.complex128)

        for n in range(cfg.Nc):
            for l in range(cfg.Lp):
                # 量化实部
                real_part = user_p['gains_all_carriers'][n, l].real
                real_idx = gain_quantizer.quantize(real_part)
                # 量化虚部
                imag_part = user_p['gains_all_carriers'][n, l].imag
                imag_idx = gain_quantizer.quantize(imag_part)

                # 反量化以在BS端重构
                dequantized_real = gain_quantizer.dequantize(real_idx)
                dequantized_imag = gain_quantizer.dequantize(imag_idx)
                q_gains_all_carriers[n, l] = dequantized_real + 1j * dequantized_imag

        quantized_params.append({
            'angle_indices': user_p['angle_indices'],  # 角度索引直接反馈
            'gains_all_carriers': q_gains_all_carriers
        })
    return quantized_params