# Attention-Driven CNN-Transformer Fusion Network for End-to-End Hybrid Precoding in FDD Massive MIMO-OFDM

## 项目简介

本仓库实现了一种基于注意力驱动的 CNN-Transformer 融合网络，用于 FDD 大规模 MIMO-OFDM 系统中的端到端混合预编码。目标是联合优化导频传输、CSI 反馈与混合预编码，以最大化系统频谱效率。

## 论文背景

论文《Attention-Driven CNN-Transformer Fusion Network Architecture for End-to-End Hybrid Precoding in FDD Massive MIMO-OFDM Systems》指出：

- FDD 大规模 MIMO-OFDM 系统中，CSI 反馈开销极大，是性能瓶颈。
- 现有数据驱动 CSI 压缩方法大多基于 CNN，受限于局部感受野，难以捕获高维信道的全局相关性。
- 论文提出将 CNN 与 Transformer 融合的网络，利用注意力机制增强 CSI 压缩，并在基站端同时学习本地与全局通道结构。

## 本仓库实现

当前代码实现了论文的核心思想，并结合实际训练流程：

- UE 端：基于注意力门控的 CNN 卷积网络（包含 CBAM 空间/通道注意力机制）对 CSI 进行压缩。
- BS 端：使用 RepVGG 风格卷积模块提取局部特征，再经 Transformer 编码器捕获全局频域相关性，用于生成混合预编码矩阵。
- 端到端训练：`train.py` 负责加载信道数据、训练 UE/BS 两端网络、计算频谱效率损失并保存最优模型。

## 关键架构

### UE 端（CSI 压缩）

- `Network_FDD_CBAM.py` 中的 `DNN_US_RF_OFDM` 实现用户端网络。
- 输入：多子载波复数 CSI 的实部与虚部表示。
- 主要模块：
  - 多天线导频映射
  - 复数卷积与 CBAM 残差块
  - 门控特征单元
  - 量化层生成二进制反馈码字

### BS 端（混合预编码生成）

- `Network_FDD_CBAM.py` 中的 `DNN_BS_hyb_OFDM` 实现基站端网络。
- 输入：UE 端压缩后的比特流。
- 主要模块：
  - 反量化层
  - 全连接特征提取
  - RepVGG 风格卷积块
  - Transformer 编码器
  - 生成射频与基带复数预编码矩阵

### 端到端损失

- `MyLoss_OFDM` 作为自定义频谱效率损失函数。
- 直接基于信道矩阵和生成的预编码器计算信号与干扰，优化最终系统频谱效率。

## 数据与训练

### 数据生成

- 信道数据生成脚本：`channel.py`。
- 生成的训练/测试数据保存在 `data/` 目录下，文件名形如：
  - `H_train_UPA{N}Lp_1.pt`
  - `H_train_UPA{N}Lp_2.pt`
  - `H_test_UPA{N}Lp.pt`
- 数据格式为 PyTorch 张量，包含用户数、接收天线、子载波、复数通道实虚部。

### 训练入口

- 主要训练脚本：`train.py`
- 处理流程：
  1. 加载训练/测试信道数据
  2. 初始化 UE/BS 两端网络
  3. 使用 Adam 优化器进行迭代训练
  4. 每轮评估测试集频谱效率
  5. 保存最佳模型到 `saved_models/`

### 默认训练参数

- 天线配置：`Nc=32`, `Nt=64`, `Nr=1`
- 用户数：`K=2`
- 子载波数、反馈比特等参数可在 `train.py` 中修改
- 反馈比特集合 `B_values = [1, 3, 16, 24, 32, 48, 64]`

## 运行示例

```bash
python train.py
```

如果需要使用其他模型变体，可修改 `train.py` 中的导入模块，例如：

```python
import Network_FDD as FDD
# 或者
import Network_FDD_f as FDD
```

## 目录与文件说明

- `train.py`：训练主脚本，加载数据并执行端到端训练
- `Network_FDD_CBAM.py`：核心实现，包含 UE/BS 网络、CBAM 模块、RepVGG 与 Transformer 结构
- `Network_FDD.py` / `Network_FDD_f.py` / `Network_FDD_Conformer.py`：不同网络变体实现，可用于对比实验
- `channel.py`：信道生成与数据处理代码
- `convert_to_npy.py`：用于将 `.pt` 数据转换为 `.npy` 或其他格式的辅助脚本
- `MO-BH/`：基线仿真。详见仓库[MO-AltMin](https://github.com/Lxlorange/MO-AltMin)
- `saved_models/`：训练得到的模型权重文件
- `data/`：信道样本数据集

## 依赖环境

建议安装：

- Python 3.x
- PyTorch
- NumPy
- SciPy
- matplotlib
- torchvision

## 说明

本项目聚焦于 FDD 大规模 MIMO-OFDM 的端到端混合预编码，目标是通过注意力驱动的网络结构压缩 CSI 并生成高效混合预编码矩阵，显著提升频谱效率。