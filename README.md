## transformer机制修改记录

### 1.init
使用`Transformer+门控.txt`源代码，运行到epoch=4时Test SE仍为7左右，隧停止。

### 2.添加层数

```python
class DNN_BS_hyb_OFDM(nn.Module):
    def __init__(self, parm_set):
        ...
        self.full_transformer = FullTransformerBlock(
            feature_dim=feature_dim,
            num_heads=4, //把8改为4
            dim_feedforward=feature_dim * 4, # FFN中间层维度, 4倍是常见设置
            num_layers=3, //把2改3
            dropout=0.1
        )
        ...
```
结果：增长显著
![img.png](assets/img.png)

### 3.增加输入特征维度

```python
class DNN_BS_hyb_OFDM(nn.Module):
    def __init__(self, parm_set):
        ...
        feature_dim = 32 # 原为2 * K * K
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
            num_layers=3,
            dropout=0.1
        )
        ...
```

结果：比2略长

```txt
Epoch: 0 | Time: 0:01:07.068966 | Train SE: 5.943 | Test SE: 7.334

<Figure size 640x480 with 1 Axes>
Model saved!
Epoch: 1 | Time: 0:01:06.667587 | Train SE: 7.556 | Test SE: 7.881
Model saved!
Epoch: 2 | Time: 0:01:07.389920 | Train SE: 9.482 | Test SE: 10.553
Model saved!
Epoch: 3 | Time: 0:01:07.408841 | Train SE: 10.837 | Test SE: 11.164

<Figure size 640x480 with 1 Axes>
Model saved!
Epoch: 4 | Time: 0:01:07.822202 | Train SE: 11.323 | Test SE: 11.576
Model saved!
Epoch: 5 | Time: 0:01:07.534253 | Train SE: 11.567 | Test SE: 11.691
Model saved!
Epoch: 6 | Time: 0:01:08.482000 | Train SE: 11.737 | Test SE: 11.953

<Figure size 640x480 with 1 Axes>
Model saved!
Epoch: 7 | Time: 0:01:08.921774 | Train SE: 11.892 | Test SE: 12.022
Model saved!
Epoch: 8 | Time: 0:01:08.287940 | Train SE: 12.002 | Test SE: 12.160
Model saved!
Epoch: 9 | Time: 0:01:08.221184 | Train SE: 12.126 | Test SE: 12.155

<Figure size 640x480 with 1 Axes>
Epoch: 10 | Time: 0:01:08.594511 | Train SE: 12.202 | Test SE: 12.299
Model saved!
Epoch: 11 | Time: 0:01:08.852681 | Train SE: 12.308 | Test SE: 12.407
Model saved!
Epoch: 12 | Time: 0:01:08.501894 | Train SE: 12.390 | Test SE: 12.508

<Figure size 640x480 with 1 Axes>
Model saved!
Epoch: 13 | Time: 0:01:09.078835 | Train SE: 12.415 | Test SE: 12.484
Epoch: 14 | Time: 0:01:08.856776 | Train SE: 12.480 | Test SE: 12.599
Model saved!
```
