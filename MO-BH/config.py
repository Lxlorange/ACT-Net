import numpy as np

# --- 系统参数---
# 基站(BS)天线数 (M = Ny * Nz)
Ny = 8
Nz = 8
M = Ny * Nz

# 同时服务的用户数(K)，也等于RF链数
K = 2

# OFDM子载波数
Nc = 32

# 信道路径数
Lp = 2

# --- 物理参数 ---
# 天线间距与波长的比值 d/lambda
d_lambda_ratio = 0.5

# 总发射功率 (W)
Pt = 1

SNR_dB = 10
SNR_linear = 10**(SNR_dB / 10)

# MO-AltMin 算法迭代次数
MAX_ITER_ALTMIN = 10  # 外层交替迭代
MAX_ITER_MANIFOLD = 20 # 内层流形优化迭代