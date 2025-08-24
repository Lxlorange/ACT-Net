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


#下面六个函数，模拟了信道信息的量化与反量化。

#1&2 底层的辅助函数，用于在浮点数和其二进制表示之间进行转换。
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

# 3&4 用户端(PFN)的最后一层，它将网络提取出的连续特征（浮点数）量化成离散的0/1比特流，模拟了真实通信中反馈信道的比特约束
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

# 5&6 基站端(FDD-HBFN)的第一层，它接收到0/1比特流后，将其反量化为连续的浮点数值，以便后续的神经网络层进行处理。
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

#############################

#损失函数
class MyLoss_OFDM(torch.nn.Module):
    def __init__(self):
        super(MyLoss_OFDM, self).__init__()

    def forward(self, H0, out, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set

        # H0 shape: [batch, K, Nr, Nc, 2*Nt] -> [batch, K, Nc, Nr, 2*Nt]
        H = H0.permute(0, 1, 3, 2, 4)
        num = H.shape[0]

        H_real = H[..., 0:Nt]
        H_imag = H[..., Nt:2 * Nt]
        H_complex = torch.complex(H_real, H_imag)  # Shape: [batch, K, Nc, Nr, Nt]

        F_real = out[:, 0:K * Nt * Nc].reshape(num, Nc, Nt, K)
        F_imag = out[:, K * Nt * Nc:2 * K * Nt * Nc].reshape(num, Nc, Nt, K)
        F_complex = torch.complex(F_real, F_imag).permute(0, 3, 1, 2)  # Shape: [batch, K, Nc, Nt]

        R = 0
        noise_power = 1 / snr

        # 遍历每个用户
        for k in range(K):
            Hk = H_complex[:, k, :, :, :]  # Channel for user k, Shape: [batch, Nc, Nr, Nt]
            Fk = F_complex[:, k, :, :]  # Beamformer for user k, Shape: [batch, Nc, Nt]
            Fk = Fk.unsqueeze(-1)  # Shape: [batch, Nc, Nt, 1]

            # 计算信号项: H_k * F_k
            signal_cov = torch.matmul(Hk, Fk)
            signal_cov = torch.matmul(signal_cov, signal_cov.mH)  # .mH is conjugate transpose

            # 计算干扰项
            interference_cov = torch.zeros(num, Nc, Nr, Nr, device=H0.device, dtype=torch.complex64)
            for j in range(K):
                if k != j:
                    Fj = F_complex[:, j, :, :].unsqueeze(-1)
                    inter_term = torch.matmul(Hk, Fj)
                    interference_cov += torch.matmul(inter_term, inter_term.mH)

            # 计算总协方差矩阵并计算速率
            # R = log2(det(I + Q_interference^-1 * Q_signal))
            # I + interference_cov / noise_power
            Q_inter_inv = torch.inverse(
                interference_cov + noise_power * torch.eye(Nr, device=H0.device).unsqueeze(0).unsqueeze(0))
            log_det_term = torch.log2(torch.det(
                torch.eye(Nr, device=H0.device).unsqueeze(0).unsqueeze(0) + torch.matmul(Q_inter_inv, signal_cov)).real)
            R += torch.sum(log_det_term)

        R = -R / num / Nc
        return R

#############################
# 激活函数mish
# 被认为是ReLU激活函数的一个更平滑、性能更好的替代品。、
# 论文中提到在每个隐藏层中使用它来为网络提供非线性映射能力，代码提供了函数式和模块式两种实现
def mish(x):
    return x * (torch.tanh(F.softplus(x)))
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

# 深度残差模块
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

# 门控特征单元
class GatedFeatureUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.Sigmoid())
        self.transform = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.GELU())

    def forward(self, x):
        return self.gate(x) * self.transform(x)

# 一个自定义的“复数卷积块”
# 神经网络的标准卷积层通常只能处理实数。
# 这个模块通过一个巧妙的方式来处理复数信号：它将输入的复数信号（即接收到的导频）分解为实部和虚部，
# 然后用两个独立的卷积核分别对它们进行卷积，最后再将结果进行交叉组合。
class ComplexConvBlock(nn.Module):
    def __init__(self, Nc, L):
        super().__init__()
        self.real_conv = nn.Conv1d(2, 2, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, Nc, L = x.shape
        x = x.view(B, 2, -1)
        x = self.real_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(B, 2, Nc, L)
        return x

# 用户端网络 - PFN
# 这个类完整地实现了论文FDD框架图中的HPN和PFN两个模块的功能。它是运行在用户设备(UE)上的神经网络。
class DNN_US_RF_OFDM(nn.Module):
    def __init__(self, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        super().__init__()
        global L
        L = 32
        self.pilot = nn.Linear(Nt, L, bias=False)
        self.res = RES_BLOCK([2 * L * Nr, 256, 512])
        self.FC2 = nn.Linear(32 * Nc * L, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        self.gated_fc3 = GatedFeatureUnit(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.FC4 = nn.Linear(512, B)
        self.bn4 = nn.BatchNorm1d(B)
        self.QL = QuantizationLayer(1)
        self.complex_conv = ComplexConvBlock(Nc, L)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 在 DNN_US_RF_OFDM 类中
    def forward(self, h, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        device = h.device
        num = h.shape[0]  # batch*K

        F_real = torch.cos(self.pilot.weight) / sqrt(Nt) * sqrt(K)
        F_imag = torch.sin(self.pilot.weight) / sqrt(Nt) * sqrt(K)
        F_complex = torch.complex(F_real, F_imag)  # Shape: [L, Nt]

        # 2. 正确处理多天线(MIMO)的信道数据
        # h的输入形状是 [batch*K, Nr, Nc, 2*Nt]
        h_real = h[..., 0:Nt]
        h_imag = h[..., Nt:2 * Nt]
        h_complex = torch.complex(h_real, h_imag)  # Shape: [batch*K, Nr, Nc, Nt]

        # 3. 模拟多天线接收过程 (矩阵乘法)
        L_complex = torch.einsum('binc,cl->binl', h_complex, F_complex.mH)

        # 4. 添加噪声
        noise = (torch.randn_like(L_complex) + 1j * torch.randn_like(L_complex)) / sqrt(2 * snr)
        L_complex = L_complex + noise

        # 5. 重塑数据以送入后续的实数CNN网络
        # 将Nr和L维度合并到"通道"维度
        # [B,Nr,L,Nc] -> [B, 2*Nr*L, Nc]
        L_real = L_complex.real.permute(0, 1, 3, 2)  # -> [batch*K, Nr, L, Nc]
        L_imag = L_complex.imag.permute(0, 1, 3, 2)  # -> [batch*K, Nr, L, Nc]
        x = torch.cat([L_real, L_imag], dim=1)  # -> [batch*K, 2*Nr, L, Nc]
        x = x.reshape(num, 2 * Nr * L, Nc, 1)  # -> [batch*K, 2*Nr*L, Nc, 1] - This is the correct input for the CNN
        x = self.res(x)
        x = x.reshape(num, -1)

        x = self.FC2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.gated_fc3(x)
        x = self.bn3(x)
        x = self.FC4(x)
        x = self.bn4(x)
        x = torch.sigmoid(x)
        x = self.QL(x)
        return x

#####################################


# 多尺度特征增强器  允许网络在同一层级上捕获不同感受野的特征
class MultiScaleFeatureEnhancer(nn.Module):
    def __init__(self, input_dim):
        super(MultiScaleFeatureEnhancer, self).__init__()
        self.scale1 = nn.Linear(input_dim, input_dim)
        self.scale2 = nn.Linear(input_dim, input_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.act = Mish()

    def forward(self, x):
        out1 = self.scale1(x)
        out2 = self.scale2(x)
        out = out1 + out2
        out = self.bn(out)
        out = self.act(out)
        return out

# 源自“RepVGG”架构的高性能卷积模块
class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels))
            self.rbr_1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                                         nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'): return self.rbr_reparam(inputs)
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out

    def reparameterize(self):
        if hasattr(self, 'rbr_reparam'): return
        kernel, bias = self._get_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.rbr_dense[0].in_channels, self.rbr_dense[0].out_channels,
                                     self.rbr_dense[0].kernel_size, self.rbr_dense[0].stride, self.rbr_dense[0].padding,
                                     bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters(): para.detach_()
        self.__delattr__('rbr_dense');
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'): self.__delattr__('rbr_identity')
        self.deploy = True

    def _get_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _fuse_bn_tensor(self, branch):
        if branch is None: return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel, bn = branch[0].weight, branch[1]
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.rbr_dense[0].in_channels
                kernel_value = torch.zeros((input_dim, input_dim, 3, 3), dtype=torch.float32)
                for i in range(input_dim): kernel_value[i, i, 1, 1] = 1
                self.id_tensor = kernel_value.to(branch.weight.device)
            kernel, bn = self.id_tensor, branch
        running_mean, running_var, gamma, beta, eps = bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

# Transformer编码器
class PositionalEncoding(nn.Module):
    """为序列添加位置编码，使其感知顺序"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
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


# 基站网络 - FDD-HBFN
# 这个类实现了论文FDD框架图中的FDD-HBFN模块。它是运行在基站(BS)上的神经网络。
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
        self.msfe = MultiScaleFeatureEnhancer(1024)
        self.FC3 = nn.Linear(1024, 2 * K * K * Nc + K * Nt)
        self.bn3 = nn.BatchNorm1d(2 * K * K * Nc)
        self.mish3 = Mish()
        feature_dim = 2 * K * K

        self.repvgg_block = nn.Sequential(
            RepVGGBlock(in_channels=2 * K * K, out_channels=feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            Mish()
        )
        self.conv = nn.Conv2d(
            in_channels=feature_dim, out_channels=feature_dim, kernel_size=(5, 1), stride=1, padding=(2, 0)
        )
        self.full_transformer = FullTransformerBlock(
            feature_dim=feature_dim,
            num_heads=4,
            dim_feedforward=feature_dim * 4,  # FFN中间层维度, 4倍是常见设置
            num_layers=4,
            dropout=0.1
        )



    def forward(self, x, parm_set):
        Nc, Nt, Nr, snr, B, K = parm_set
        device = x.device

        x = self.DQL(x) - 0.5
        x = self.FC1(x);
        x = self.bn1(x);
        x = self.mish1(x)
        x = self.FC2(x);
        x = self.bn2(x);
        x = self.mish2(x)
        x = self.msfe(x)
        x_ini = self.FC3(x)

        RF_real = torch.cos(x_ini[:, 2 * K * K * Nc:(2 * K * K * Nc + K * Nt)]).reshape(-1, Nt, K) / torch.sqrt(
            torch.tensor(Nt, device=device))
        RF_imag = torch.sin(x_ini[:, 2 * K * K * Nc:(2 * K * K * Nc + K * Nt)]).reshape(-1, Nt, K) / torch.sqrt(
            torch.tensor(Nt, device=device))

        x = x_ini[:, 0:2 * K * K * Nc]
        x = self.bn3(x)
        x = self.mish3(x)

        x = x.reshape(-1, 2 * K * K, Nc, 1)
        x = self.repvgg_block(x)
        x = self.conv(x)

        # --- ✨ 修改点 2: 调整数据流以适应新的Transformer ---
        # 1. 准备输入: from [batch, features, Nc, 1] to [batch, Nc, features]
        x_in = x.squeeze(-1).permute(0, 2, 1)

        # 2. 调用完整的Transformer模块
        x_out = self.full_transformer(x_in)  # output is [batch, Nc, features]

        # 3. 恢复原始维度顺序以便后续处理
        x = x_out.permute(0, 2, 1).unsqueeze(-1)  # -> [batch, features, Nc, 1]
        # --- 修改结束 ---

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