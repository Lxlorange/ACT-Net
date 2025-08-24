# config.py

import numpy as np

# --- 系统参数---
# 基站(BS)天线数 (M = Ny * Nz)
Ny = 8
Nz = 8
M = Ny * Nz

# 用户侧接收天线数 (论文明确为单天线)
Nr = 1

# 同时服务的用户数(K)，也等于RF链数
K = 2

# OFDM子载波数
Nc = 32

# 信道路径数 (Sparsity Level)
Lp = 2

# --- 物理参数 ---
d_lambda_ratio = 0.5
Pt = 1
SNR_dB = 10
SNR_linear = 10**(SNR_dB / 10)

# --- 算法迭代参数 ---
MAX_ITER_ALTMIN = 10
MAX_ITER_MANIFOLD = 20
MAX_ITER_WMMSE = 10       # 新增: WMMSE算法迭代次数

# --- 信道估计与反馈参数 ---
# FDD下行导频OFDM符号数 (来自论文图11a标题)
Q = 8

# 每个稀疏路径参数的量化比特数
# 论文提到5个参数: Re(alpha), Im(alpha), theta, phi, tau
# 这里我们简化为量化复数增益的实部和虚部
FEEDBACK_BITS_PER_PARAM = 4