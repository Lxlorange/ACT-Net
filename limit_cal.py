# _*_ coding : utf-8 _*_
# @Time :  19:55
# @Author : Lxl
# @File ： limit_cal
# @ProjectName : OFDM
import torch
import numpy as np


def calculate_theoretical_upper_bound(H_perfect, P_t, snr, K, Nr, Nt, Nc):
    """
    计算多用户MIMO-OFDM系统在完美CSI和全数字迫零波束赋形下的理论Sum Rate上限。

    Args:
        H_perfect (torch.Tensor): 完美的信道数据集，形状为 [batch_size, K, Nr, Nc, 2*Nt]。
        P_t (float): 总发射功率。
        snr (float): 线性信噪比 (例如 SNR_dB=10 -> snr=10)。
        K (int): 用户数。
        Nr (int): 每个用户的接收天线数。
        Nt (int): 基站的发射天线数。
        Nc (int): 子载波数量。

    Returns:
        float: 理论上的平均Sum Rate (bps/Hz)。
    """

    # 0. 准备工作
    batch_size = H_perfect.shape[0]
    device = H_perfect.device

    # 将输入的实数信道数据转换为复数张量
    # 形状: [batch, K, Nr, Nc, Nt]
    H_complex = torch.complex(H_perfect[..., 0:Nt], H_perfect[..., Nt:2 * Nt])

    total_sum_rate_all_samples = 0.0

    # 对batch中的每个信道样本进行处理
    # (实际中这可以并行化，但为清晰起见，我们逐个样本处理)
    for i in range(batch_size):
        sum_rate_per_sample = 0.0

        # 1. 对每一个子载波独立计算
        for n in range(Nc):
            # 提取当前子载波n的所有用户信道
            # 形状: [K, Nr, Nt]
            H_subcarrier_all_users = H_complex[i, :, :, n, :]

            # 2. 将所有用户信道堆叠成一个大的总信道矩阵 H_total
            # 形状: [K*Nr, Nt]
            H_total = H_subcarrier_all_users.reshape(K * Nr, Nt)

            # 3. 计算ZF预编码矩阵 W = H^H * (H * H^H)^-1
            # 我们使用更稳健的伪逆 torch.linalg.pinv(H) 来计算
            # pinv(H) 的形状是 [Nt, K*Nr]
            W_zf = torch.linalg.pinv(H_total)

            # 4. 功率归一化
            # F_sigma_sq = torch.sum(torch.abs(W_zf)**2) # 弗罗贝尼乌斯范数的平方
            F_sigma_sq = torch.real(torch.trace(torch.matmul(W_zf, W_zf.mH)))
            beta = torch.sqrt(P_t / F_sigma_sq)
            F = beta * W_zf  # 最终的预编码矩阵, 形状: [Nt, K*Nr]

            sum_rate_per_subcarrier = 0.0

            # 5. 为每个用户计算速率并求和
            for k in range(K):
                # 提取用户k的信道矩阵 Hk, 形状: [Nr, Nt]
                Hk = H_subcarrier_all_users[k, :, :]

                # 提取用户k的波束赋形矩阵 Fk, 形状: [Nt, Nr]
                start_col = k * Nr
                end_col = (k + 1) * Nr
                Fk = F[:, start_col:end_col]

                # 6. 使用MIMO信道容量公式计算速率
                # R_k = log2(det(I + (SNR/Nr) * Hk * Fk * Fk^H * Hk^H))
                # 注意: 这里的snr已经是 P_t / sigma^2, 包含了功率
                # 噪声协方差为 I, 信号协方差为 Hk*Fk*Fk^H*Hk^H
                # 公式为 log2(det(I + (1/noise_power) * signal_cov))

                signal_cov = torch.matmul(Hk, Fk)
                signal_cov = torch.matmul(signal_cov, signal_cov.mH)  # .mH is conjugate transpose

                identity_matrix = torch.eye(Nr, device=device, dtype=torch.complex64)

                # snr = P_t / noise_power, 所以 1/noise_power = snr / P_t
                rate_k = torch.log2(torch.det(identity_matrix + (snr / P_t) * signal_cov).real)
                sum_rate_per_subcarrier += rate_k

            sum_rate_per_sample += sum_rate_per_subcarrier

        # 7. 对所有子载波的速率取平均
        total_sum_rate_all_samples += sum_rate_per_sample / Nc

    # 8. 对所有样本的速率取平均
    return (total_sum_rate_all_samples / batch_size).item()


if __name__ == '__main__':
    # --- 设定与您实验一致的参数 ---
    Nt = 64 # 发射天线数 (12x12)
    Nr = 1  # 单个用户的接收天线数
    K = 2  # 用户数
    Nc = 32  # 子载波数量
    SNR_dB = 10 # 信噪比 (dB)

    # --- 转换参数 ---
    P_t = 1.0  # 假设总发射功率为1 (通常在学术仿真中归一化)
    snr_linear = 10 ** (SNR_dB / 10.0)  # 将dB转换为线性值

    # --- 生成一批随机的、符合您数据格式的完美信道数据用于测试 ---
    batch_size = 50  # 我们可以只用少量样本来估算理论值
    # 形状: [batch_size, K, Nr, Nc, 2*Nt]
    # (randn返回标准正态分布，除以sqrt(2)是为了使复数信道的总功率为1)
    h_real_part = torch.randn(batch_size, K, Nr, Nc, Nt) / np.sqrt(2)
    h_imag_part = torch.randn(batch_size, K, Nr, Nc, Nt) / np.sqrt(2)
    H_perfect_sample = torch.cat([h_real_part, h_imag_part], dim=-1)

    # 将数据移动到GPU（如果可用）
    if torch.cuda.is_available():
        H_perfect_sample = H_perfect_sample.cuda()
        print("正在使用GPU进行计算...")
    else:
        print("正在使用CPU进行计算...")

    # --- 调用函数计算理论上限 ---
    print("\n正在计算理论Sum Rate上限...")
    theoretical_max_sum_rate = calculate_theoretical_upper_bound(
        H_perfect=H_perfect_sample,
        P_t=P_t,
        snr=snr_linear,
        K=K,
        Nr=Nr,
        Nt=Nt,
        Nc=Nc
    )

    print("-" * 40)
    print(f"系统参数:")
    print(f"  基站天线数 (Nt): {Nt}")
    print(f"  用户天线数 (Nr): {Nr}")
    print(f"  用户数 (K): {K}")
    print(f"  信噪比 (SNR): {SNR_dB} dB")
    print("-" * 40)
    print(f"全数字ZF波束赋形下的理论Sum Rate上限约为: {theoretical_max_sum_rate:.4f} bps/Hz")
    print("-" * 40)