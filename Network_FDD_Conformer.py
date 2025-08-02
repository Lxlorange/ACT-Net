import math
import torch
import os

from torch.nn import TransformerEncoderLayer, TransformerEncoder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
import torch.nn.functional as F
import torchvision
import numpy as np
from math import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
from IPython import display
import torch.utils.data as Data
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from scipy.linalg import block_diag
import datetime
from torch.nn.utils import *

# --- 系统参数 ---
Nc = 32
Nt = 64
Nr = 1
L = 8
K = 2
SNR_dB = 10
snr = 10 ** (SNR_dB / 10) / K


# ----------------------------------------------------------------------
# --- 1. 辅助函数、量化层 和 损失函数 (与之前版本相同) ---
# ----------------------------------------------------------------------

def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1, device=Num.device).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 0].shape, device=Bit.device)
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_bit = grad_output.repeat_interleave(ctx.constant, dim=1)
        return grad_bit, None


class QuantizationLayer(nn.Module):
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        return Quantization.apply(x, self.B)


class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        return Dequantization.apply(x, self.B)


class MyLoss_OFDM(torch.nn.Module):
    def __init__(self):
        super(MyLoss_OFDM, self).__init__()

    def forward(self, H0, out, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        H = H0.permute(0, 2, 1, 3)
        num = out.shape[0]
        H_real = H[:, :, :, 0:Nt]
        H_imag = H[:, :, :, Nt:2 * Nt]
        Hs = torch.zeros([num, Nc, K * 2, Nt * 2], device=H0.device)
        Hs[:, :, 0:K, 0:Nt] = H_real
        Hs[:, :, K:2 * K, Nt:2 * Nt] = H_real
        Hs[:, :, 0:K, Nt:2 * Nt] = H_imag
        Hs[:, :, K:2 * K, 0:Nt] = -H_imag
        F = torch.zeros([num, Nc, Nt * 2, K * 2], device=H0.device)
        F_real_part = out[:, 0:K * Nt * Nc].reshape(num, Nc, Nt, K)
        F_imag_part = out[:, K * Nt * Nc:2 * K * Nt * Nc].reshape(num, Nc, Nt, K)
        F[:, :, 0:Nt, 0:K] = F_real_part
        F[:, :, Nt:2 * Nt, K:2 * K] = F_real_part
        F[:, :, 0:Nt, K:2 * K] = F_imag_part
        F[:, :, Nt:2 * Nt, 0:K] = -F_imag_part
        R = 0
        Hk = torch.matmul(Hs, F)
        noise = 1 / snr
        for i in range(K):
            signal = Hk[:, :, i, i] ** 2 + Hk[:, :, i, i + K] ** 2
            interference = torch.zeros(num, Nc, device=H0.device)
            for j in range(K):
                if j != i:
                    interference = interference + Hk[:, :, i, j] ** 2 + Hk[:, :, i, j + K] ** 2
            SINR = signal / (noise + interference)
            R = R + torch.sum(torch.log2(1 + SINR))
        R = -R / num / Nc
        return R


def mish(x):
    return x * (torch.tanh(F.softplus(x)))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(- 2 * pi * 1J / N)
    W = np.power(omega, i * j) / sqrt(N)
    return np.mat(W)


# ----------------------------------------------------------------------
# --- 2. 用户端网络 和 相关模块 (与之前版本相同) ---
# ----------------------------------------------------------------------

class RES_BLOCK(nn.Module):
    def __init__(self, channel_list):
        super(RES_BLOCK, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[1], kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(channel_list[1]), Mish())
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[2], kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(channel_list[2]), Mish())
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_list[2], channel_list[0], kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(channel_list[0]))

    def forward(self, x_ini):
        x = self.conv1(x_ini)
        x = self.conv2(x)
        x = self.conv3(x)
        x = mish(x + x_ini)
        return x


class GatedFeatureUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.Sigmoid())
        self.transform = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.GELU())

    def forward(self, x):
        return self.gate(x) * self.transform(x)


class DNN_US_RF_OFDM(nn.Module):
    def __init__(self, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        super().__init__()
        global L
        L = 32
        self.pilot = nn.Linear(Nt, L, bias=False)
        self.res = RES_BLOCK([2 * L, 256, 512])
        self.FC2 = nn.Linear(2 * Nc * L, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        self.gated_fc3 = GatedFeatureUnit(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.FC4 = nn.Linear(512, B)
        self.bn4 = nn.BatchNorm1d(B)
        self.QL = QuantizationLayer(1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, h, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        device = h.device
        h_real = h[:, :, 0:Nt].reshape(-1, Nc, Nt, 1)
        h_imag = h[:, :, Nt:2 * Nt].reshape(-1, Nc, Nt, 1)
        F_real = torch.cos(self.pilot.weight) / sqrt(Nt) * sqrt(K)
        F_imag = torch.sin(self.pilot.weight) / sqrt(Nt) * sqrt(K)
        L_real = (torch.matmul(F_real, h_real) - torch.matmul(F_imag, h_imag)).reshape(-1, 1, Nc, L)
        L_imag = (torch.matmul(F_real, h_imag) + torch.matmul(F_imag, h_real)).reshape(-1, 1, Nc, L)
        L_sum = torch.cat((L_real, L_imag), 1)
        num = h.shape[0]
        noise = torch.randn(num, 2, Nc, L, device=device) / sqrt(2 * snr)
        L_sum = L_sum + noise
        x = L_sum.transpose(1, 2).reshape(-1, 2 * L, Nc, 1)
        x = self.res(x)
        x = x.reshape(-1, 2 * L * Nc)
        x = self.FC2(x);
        x = self.bn2(x);
        x = self.relu2(x)
        x = self.gated_fc3(x);
        x = self.bn3(x)
        x = self.FC4(x);
        x = self.bn4(x)
        x = torch.sigmoid(x)
        x = self.QL(x)
        return x


# ----------------------------------------------------------------------
# --- 3. 新增模块: Conformer 架构所需的所有组件 ---
# ----------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """为序列添加位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x 的 shape: [batch_size, seq_len, d_model] """
        # 注意：Conformer模块的输入是(B,S,D)，所以位置编码也需要适配
        x = x + self.pe[:x.size(1)].squeeze(1)
        return self.dropout(x)


class Swish(nn.Module):
    """ Swish激活函数 """

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvolutionModule(nn.Module):
    """ Conformer中的卷积模块 """

    def __init__(self, channels, kernel_size=15):
        super(ConvolutionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.glu = nn.GLU(dim=1)  # 门控线性单元
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1,
                                        padding=(kernel_size - 1) // 2, groups=channels)
        self.batch_norm = nn.BatchNorm1d(channels)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """ x 的 shape: [batch, seq_len, channels] """
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # -> [batch, channels, seq_len]
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # -> [batch, seq_len, channels]
        return residual + x


class SelfAttentionModule(nn.Module):
    """ Conformer中的多头自注意力模块 """

    def __init__(self, d_model, nhead, dropout=0.1):
        super(SelfAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """ x 的 shape: [batch, seq_len, d_model] """
        residual = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        return residual + x


class FeedForwardModule(nn.Module):
    """ Conformer中的前馈网络模块 """

    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + x


class ConformerBlock(nn.Module):
    """ 完整的Conformer块: FFN -> Attention -> Conv -> FFN """

    def __init__(self, d_model, dim_feedforward, nhead, kernel_size=15, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ffn1 = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.attention = SelfAttentionModule(d_model, nhead, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size)
        self.ffn2 = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Conformer的结构是 FFN -> Attention -> Conv -> FFN
        # 每个模块内部都有自己的残差连接
        x = self.ffn1(x)
        x = self.attention(x)
        x = self.conv(x)
        x = self.ffn2(x)
        # 最终输出前再做一次LayerNorm
        x = self.layer_norm(x)
        return x


# ----------------------------------------------------------------------
# --- 4. 基站端网络: 采用全新的 Conformer 架构 ---
# ----------------------------------------------------------------------

class DNN_BS_hyb_OFDM(nn.Module):
    def __init__(self, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        super(DNN_BS_hyb_OFDM, self).__init__()

        # --- 前端全连接层 (保持不变) ---
        self.DQL = DequantizationLayer(1)
        self.FC1 = nn.Linear(K * B, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.mish1 = Mish()
        self.FC2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.mish2 = Mish()
        self.FC3 = nn.Linear(1024, 2 * K * K * Nc + K * Nt)
        self.bn3 = nn.BatchNorm1d(2 * K * K * Nc)
        self.mish3 = Mish()

        # --- ✨ 架构修改点: 使用 Conformer 栈 ---
        # 1. 定义新的特征维度
        feature_dim = 32  # 可以调整, e.g., 32, 64

        # 2. 定义一个初始卷积层，将通道数映射到我们想要的feature_dim
        self.initial_conv = nn.Conv2d(
            in_channels=2 * K * K, out_channels=feature_dim, kernel_size=(5, 1), stride=1, padding=(2, 0)
        )

        # 3. 引入位置编码
        self.pos_encoder = PositionalEncoding(d_model=feature_dim, dropout=0.1)

        # 4. 堆叠多个Conformer块
        num_conformer_layers = 4  # 可调整的超参数, e.g., 2, 3, 4
        self.conformer_stack = nn.Sequential(*[
            ConformerBlock(
                d_model=feature_dim,
                dim_feedforward=feature_dim * 4,
                nhead=4,
                kernel_size=15,
                dropout=0.1
            ) for _ in range(num_conformer_layers)
        ])

        # 5. 定义最终用于拆分的线性层
        self.final_linear = nn.Linear(feature_dim, 2 * K * K)

    def forward(self, x, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        device = x.device

        # --- 前端全连接层的前向传播 (保持不变) ---
        x = self.DQL(x) - 0.5
        x = self.FC1(x);
        x = self.bn1(x);
        x = self.mish1(x)
        x = self.FC2(x);
        x = self.bn2(x);
        x = self.mish2(x)
        x_ini = self.FC3(x)

        # --- RF 部分 (保持不变) ---
        RF_real = torch.cos(x_ini[:, 2 * K * K * Nc:(2 * K * K * Nc + K * Nt)]).reshape(-1, Nt, K) / torch.sqrt(
            torch.tensor(Nt, dtype=torch.float32, device=device))
        RF_imag = torch.sin(x_ini[:, 2 * K * K * Nc:(2 * K * K * Nc + K * Nt)]).reshape(-1, Nt, K) / torch.sqrt(
            torch.tensor(Nt, dtype=torch.float32, device=device))

        x = x_ini[:, 0:2 * K * K * Nc]
        x = self.bn3(x)
        x = self.mish3(x)
        x = x.reshape(-1, 2 * K * K, Nc, 1)

        # --- ✨ 新的数据流 ---
        # 1. 通过初始卷积层，将通道数调整为feature_dim
        x = self.initial_conv(x)  # -> [batch, feature_dim, Nc, 1]

        # 2. 准备输入给Conformer: from [b, d, n, 1] to [b, n, d]
        x = x.squeeze(-1).permute(0, 2, 1)

        # 3. 添加位置编码
        x = self.pos_encoder(x)

        # 4. 通过Conformer栈
        x = self.conformer_stack(x)  # -> [b, n, d]

        # 5. 通过最终线性层，映射回BB所需的维度
        x = self.final_linear(x)  # -> [b, n, 2*K*K]

        # 6. 维度调整以匹配后续处理
        x = x.permute(0, 2, 1).unsqueeze(-1)  # -> [b, 2*K*K, n, 1]

        # --- 后续BB矩阵计算和功率归一化部分 (保持不变) ---
        BB_real = x[:, 0:K * K, :, 0].permute(0, 2, 1).reshape(-1, Nc, K, K)
        BB_imag = x[:, K * K:2 * K * K, :, 0].permute(0, 2, 1).reshape(-1, Nc, K, K)
        BB_real = BB_real.permute(1, 0, 2, 3)
        BB_imag = BB_imag.permute(1, 0, 2, 3)

        F_real = (torch.matmul(RF_real, BB_real) - torch.matmul(RF_imag, BB_imag)).reshape(Nc, -1, K * Nt)
        F_imag = (torch.matmul(RF_imag, BB_real) + torch.matmul(RF_real, BB_imag)).reshape(Nc, -1, K * Nt)
        F_real = F_real.permute(1, 0, 2)
        F_imag = F_imag.permute(1, 0, 2)
        F = torch.cat((F_real, F_imag), 2)

        F_sigma = torch.sqrt(torch.sum(F * F, [2], keepdim=True))
        sigma2 = torch.sqrt(torch.tensor(K, dtype=torch.float32, device=device))
        F = F / (F_sigma + 1e-8) * torch.min(F_sigma.squeeze(-1), sigma2).unsqueeze(-1)

        F_real = F[:, :, 0:K * Nt].reshape(-1, K * Nt * Nc)
        F_imag = F[:, :, K * Nt:2 * K * Nt].reshape(-1, K * Nt * Nc)
        F = torch.cat((F_real, F_imag), 1)

        return F