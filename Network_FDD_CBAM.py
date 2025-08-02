import torch
import os
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
from torch.nn import TransformerEncoderLayer, TransformerEncoder
# --- 原始参数设定 (未修改) ---
Nc = 32  # number of subcarriers
N = 2  # Number of paths
Nt = 64  # Number of Antennas at the BS
Nr = 1  # Number of Antennas at the UE
L = 8  # number of pilot OFDM symbols
SNR_dB = 10  # SNR
K = 2  # number of UEs
snr = 10 ** (SNR_dB / 10) / K


# --- 辅助函数 (未修改) ---
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


# --- 损失函数 (未修改) ---
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


# --- 激活函数 (未修改) ---
def mish(x):
    return x * (torch.tanh(F.softplus(x)))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


# --- CBAM 模块 (未修改) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        bottleneck_planes = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, bottleneck_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(bottleneck_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# --- SE_RES_BLOCK (未修改) ---
class SE_RES_BLOCK(nn.Module):
    def __init__(self, channel_list, reduction=16):
        super(SE_RES_BLOCK, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[1], kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(channel_list[1]),
            Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[2], kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(channel_list[2]),
            Mish(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_list[2], channel_list[0], kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(channel_list[0]),
        )
        self.attention = CBAM(channel_list[0], reduction)

    def forward(self, x_ini):
        x = self.conv1(x_ini)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention(x)
        x = mish(x + x_ini)
        return x


def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(- 2 * pi * 1J / N)
    W = np.power(omega, i * j) / sqrt(N)
    return np.mat(W)


W = DFT_matrix(Nc)
W_real = torch.from_numpy(np.real(W)).cuda().float()
W_imag = torch.from_numpy(np.imag(W)).cuda().float()


# ===================================================================
# =================== ✨ 1. UE端编码器升级 ✨ =======================
# ===================================================================
class DNN_US_RF_OFDM(nn.Module):
    def __init__(self, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        super(DNN_US_RF_OFDM, self).__init__()

        self.pilot = nn.Linear(Nt, L, bias=False)

        # 分支1: 频域特征提取器 (保留原有的强大模块)
        self.freq_extractor = SE_RES_BLOCK([2 * L, 256, 512])

        # ✨ 分支2: 新增的轻量级时域(延迟)特征提取器
        delay_feature_channels = 8  # 定义时域分支输出的特征通道数
        self.delay_extractor = nn.Sequential(
            # 输入通道数为Nc(32)，输出为8
            nn.Conv2d(Nc, delay_feature_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(delay_feature_channels),
            Mish()
        )

        # ✨ 修改FC2的输入维度以容纳拼接后的特征
        freq_feature_size = 2 * Nc * L
        delay_feature_size = delay_feature_channels * 2 * L
        combined_feature_size = freq_feature_size + delay_feature_size

        self.FC2 = nn.Linear(combined_feature_size, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        self.FC3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.FC4 = nn.Linear(512, B)
        self.bn4 = nn.BatchNorm1d(B)
        self.QL = QuantizationLayer(1)

    def forward(self, h, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        num = h.shape[0]
        h_real = h[:, :, 0:Nt].reshape(-1, Nc, Nt, 1)
        h_imag = h[:, :, Nt:2 * Nt].reshape(-1, Nc, Nt, 1)
        F_real = torch.cos(self.pilot.weight) / sqrt(Nt) * sqrt(K)
        F_imag = torch.sin(self.pilot.weight) / sqrt(Nt) * sqrt(K)
        L_real = (torch.matmul(F_real, h_real) - torch.matmul(F_imag, h_imag)).reshape(-1, 1, Nc, L)
        L_imag = (torch.matmul(F_real, h_imag) + torch.matmul(F_imag, h_real)).reshape(-1, 1, Nc, L)
        L_sum = torch.cat((L_real, L_imag), 1)
        noise = torch.randn(num, 2, Nc, L, device=h.device) / sqrt(2 * snr)
        L_sum = L_sum + noise
        L_real = (torch.matmul(W_real, L_sum[:, 0, :, :]) - torch.matmul(W_imag, L_sum[:, 1, :, :])).reshape(-1, Nc, L)
        L_imag = (torch.matmul(W_imag, L_sum[:, 0, :, :]) + torch.matmul(W_real, L_sum[:, 1, :, :])).reshape(-1, Nc, L)
        L_sum = torch.cat((L_real, L_imag), 2)
        x = L_sum.transpose(1, 2)
        x_original = x.reshape(-1, 2 * L, Nc, 1)  # Shape: (B, 16, 32, 1)

        # ✨ --- 双分支特征提取 ---
        # 1. 频域分支 (主分支)
        x_freq = self.freq_extractor(x_original)
        x_freq_flat = x_freq.reshape(num, -1)

        # 2. 时域分支 (新分支)
        # (B, 16, 32, 1) -> (B, 32, 16, 1)
        x_permuted = x_original.permute(0, 2, 1, 3).contiguous()
        x_delay = self.delay_extractor(x_permuted)
        x_delay_flat = x_delay.reshape(num, -1)

        # 3. 拼接两种特征
        x_combined = torch.cat([x_freq_flat, x_delay_flat], dim=1)
        # --- 结束 ---

        # 使用拼接后的特征输入FC层
        x = self.FC2(x_combined)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.FC3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.FC4(x)
        x = self.bn4(x)
        x = torch.sigmoid(x)
        x = self.QL(x)
        return x


class PositionalEncoding(nn.Module):
    """为序列添加位置编码，使其感知顺序"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class FullTransformerBlock(nn.Module):
    """
    一个更稳健的、带有残差连接和位置编码的完整Transformer模块。
    """

    def __init__(self, feature_dim, num_heads, dim_feedforward, num_layers=1, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(feature_dim)
        self.pos_encoder = PositionalEncoding(d_model=feature_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # 关键：使输入格式为 (batch, seq, feature)
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        输入 x 的 shape: [batch_size, Nc, feature_dim]
        """
        # 残差连接的输入
        residual = x

        # 预归一化
        x = self.pre_norm(x)

        # 添加位置编码 (需要转换维度)
        x = x.permute(1, 0, 2)  # -> [Nc, batch_size, feature_dim]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # -> [batch_size, Nc, feature_dim]

        # 通过官方Transformer Encoder
        x = self.transformer_encoder(x)

        # ✨ 应用核心的残差连接，防止性能下降
        return residual + x

class TransformerModule(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, max_len=100):
        super(TransformerModule, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super(GatedFusion, self).__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_local, x_global):
        x_concat = torch.cat((x_local, x_global), dim=-1)
        gate = self.gate_fc(x_concat)
        x_fused = gate * x_local + (1.0 - gate) * x_global
        return self.norm(x_fused)


class DNN_BS_hyb_OFDM(nn.Module):
    def __init__(self, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        super(DNN_BS_hyb_OFDM, self).__init__()

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

        d_model = 2 * K * K

        self.local_extractor = SE_RES_BLOCK([d_model, 256, 512])

        self.global_extractor = FullTransformerBlock(
            feature_dim=d_model,  # d_model 对应 feature_dim
            num_heads=4,  # nhead 对应 num_heads
            num_layers=2,  # num_encoder_layers 对应 num_layers
            dim_feedforward=d_model * 4,
            dropout=0.1
        )

        self.fusion_module = GatedFusion(d_model)

    def forward(self, x, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        num = x.shape[0]
        x = self.DQL(x) - 0.5
        x = self.FC1(x)
        x = self.bn1(x)
        x = self.mish1(x)
        x = self.FC2(x)
        x = self.bn2(x)
        x = self.mish2(x)
        x_ini = self.FC3(x)
        RF_real = torch.cos(x_ini[:, 2 * K * K * Nc:]).reshape(num, Nt, K) / sqrt(Nt)
        RF_imag = torch.sin(x_ini[:, 2 * K * K * Nc:]).reshape(num, Nt, K) / sqrt(Nt)
        x = x_ini[:, 0:2 * K * K * Nc]
        x = self.bn3(x)
        x = self.mish3(x)
        x = x.reshape(-1, 2 * K * K, Nc, 1)

        x_local = self.local_extractor(x).squeeze(-1).permute(0, 2, 1).contiguous()
        x_global = x.squeeze(-1).permute(0, 2, 1).contiguous()
        x_global = self.global_extractor(x_global)
        x_fused = self.fusion_module(x_local, x_global)
        x = x_fused.permute(0, 2, 1).contiguous().unsqueeze(-1)
        x = x.transpose(1, 2)

        BB_real_flat = x[:, :, 0:K * K, 0]
        BB_imag_flat = x[:, :, K * K:2 * K * K, 0]
        BB_real = BB_real_flat.reshape(num, Nc, K, K)
        BB_imag = BB_imag_flat.reshape(num, Nc, K, K)
        F_real_combined = torch.einsum('bta,bnak->bntk', RF_real, BB_real) - torch.einsum('bta,bnak->bntk', RF_imag,
                                                                                          BB_imag)
        F_imag_combined = torch.einsum('bta,bnak->bntk', RF_imag, BB_real) + torch.einsum('bta,bnak->bntk', RF_real,
                                                                                          BB_imag)
        F_real_flat = F_real_combined.reshape(num, Nc, -1)
        F_imag_flat = F_imag_combined.reshape(num, Nc, -1)
        F = torch.cat((F_real_flat, F_imag_flat), 2)
        F_sigma = torch.sqrt(torch.sum(F * F, [2], keepdim=True))
        sigma2 = torch.tensor([sqrt(K)], device=x.device)
        F = F / (F_sigma + 1e-8) * torch.min(F_sigma, sigma2)
        F_real = F[:, :, 0:K * Nt].reshape(num, -1)
        F_imag = F[:, :, K * Nt:2 * K * Nt].reshape(num, -1)
        F = torch.cat((F_real, F_imag), 1)
        return F