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

```txt
Epoch: 79 | Time: 0:01:11.022755 | Train SE: 12.330 | Test SE: 12.587
The best SE is: 12.625
[ 7.12676351  9.21159353 10.60757976 10.83850009 11.02139657 11.39121189
 11.24558992 11.64600976 11.84085808 11.6913511  11.93024266 12.09318976
 12.03001854 12.01770666 11.89779222 12.10945964 12.19317183 12.17649271
 12.30879583 12.3840904  12.25690188 12.37121866 12.50987923 12.29931056
 12.42041206 12.48363984 12.39395075 12.49491434 12.59197178 12.50903807
 12.62481983 12.19721041 12.3083281  12.38593335 12.38228328 12.25494523
 12.17666302 12.22143826 12.40815835 12.28949869 12.25235896 12.3956496
 12.00135331 12.27884142 12.25185857 12.06858654 12.14730747 12.42199655
 12.39706857 12.39481435 12.28836367 12.47801752 12.53127587 12.48453534
 12.36575978 12.31589029 12.19766934 12.12308714 12.3306102  12.39959886
 12.30120487 12.29308538 12.38722796 12.26678185 12.28242912 12.46262844
 12.30168097 12.30572309 12.21192923 12.37982867 12.48691869 12.38919773
 12.44047031 12.48105645 12.54932139 12.40820775 12.486466   12.51309612
 12.4417146  12.58695462]
```
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
Epoch: 15 | Time: 0:01:08.518496 | Train SE: 12.514 | Test SE: 12.585

<Figure size 640x480 with 1 Axes>
Epoch: 16 | Time: 0:01:08.485344 | Train SE: 12.522 | Test SE: 12.519
Epoch: 17 | Time: 0:01:08.396406 | Train SE: 12.552 | Test SE: 12.594
Epoch: 18 | Time: 0:01:08.329789 | Train SE: 12.594 | Test SE: 12.624

<Figure size 640x480 with 1 Axes>
Model saved!
Epoch: 19 | Time: 0:01:08.387205 | Train SE: 12.593 | Test SE: 12.595
Epoch: 20 | Time: 0:01:08.218908 | Train SE: 12.564 | Test SE: 12.665
Model saved!
Epoch: 21 | Time: 0:01:08.577991 | Train SE: 12.583 | Test SE: 12.679

<Figure size 640x480 with 1 Axes>
Model saved!
Epoch: 22 | Time: 0:01:08.519870 | Train SE: 12.641 | Test SE: 12.709
Model saved!
Epoch: 23 | Time: 0:01:08.089200 | Train SE: 12.629 | Test SE: 12.662
Epoch: 24 | Time: 0:01:08.168721 | Train SE: 12.653 | Test SE: 12.624

<Figure size 640x480 with 1 Axes>
Epoch: 25 | Time: 0:01:10.042662 | Train SE: 12.589 | Test SE: 12.620
Epoch: 26 | Time: 0:01:10.575336 | Train SE: 12.665 | Test SE: 12.715
Model saved!
Epoch: 27 | Time: 0:01:08.426411 | Train SE: 12.524 | Test SE: 12.437

<Figure size 640x480 with 1 Axes>
Epoch: 28 | Time: 0:01:08.652444 | Train SE: 12.443 | Test SE: 12.558
Epoch: 29 | Time: 0:01:08.274207 | Train SE: 12.515 | Test SE: 12.622
Epoch: 30 | Time: 0:01:09.112688 | Train SE: 12.520 | Test SE: 12.650

<Figure size 640x480 with 1 Axes>
Epoch: 31 | Time: 0:01:10.491564 | Train SE: 12.591 | Test SE: 12.636
Epoch: 32 | Time: 0:01:11.136029 | Train SE: 12.591 | Test SE: 12.698
Epoch: 33 | Time: 0:01:10.555020 | Train SE: 12.650 | Test SE: 12.747

<Figure size 640x480 with 1 Axes>
Model saved!
Epoch: 34 | Time: 0:01:11.091176 | Train SE: 12.612 | Test SE: 12.698
Epoch: 35 | Time: 0:01:11.001747 | Train SE: 12.647 | Test SE: 12.649
Epoch: 36 | Time: 0:01:10.758029 | Train SE: 12.508 | Test SE: 12.434

<Figure size 640x480 with 1 Axes>
Epoch: 37 | Time: 0:01:10.852955 | Train SE: 12.450 | Test SE: 12.639
Epoch: 38 | Time: 0:01:10.691711 | Train SE: 12.544 | Test SE: 12.552
Epoch: 39 | Time: 0:01:11.182613 | Train SE: 12.421 | Test SE: 12.535

<Figure size 640x480 with 1 Axes>
Epoch: 40 | Time: 0:01:09.977024 | Train SE: 12.360 | Test SE: 12.416
Epoch: 41 | Time: 0:01:08.399843 | Train SE: 12.464 | Test SE: 12.613
Epoch: 42 | Time: 0:01:08.696736 | Train SE: 12.496 | Test SE: 12.447

<Figure size 640x480 with 1 Axes>
Epoch: 43 | Time: 0:01:08.446676 | Train SE: 12.427 | Test SE: 12.400
Epoch: 44 | Time: 0:01:08.214088 | Train SE: 12.471 | Test SE: 12.635
Epoch: 45 | Time: 0:01:09.218341 | Train SE: 12.533 | Test SE: 12.582

<Figure size 640x480 with 1 Axes>
Epoch: 46 | Time: 0:01:10.714668 | Train SE: 12.518 | Test SE: 12.493
Epoch: 47 | Time: 0:01:11.352685 | Train SE: 12.281 | Test SE: 12.288
Epoch: 48 | Time: 0:01:12.294331 | Train SE: 12.316 | Test SE: 12.387

<Figure size 640x480 with 1 Axes>
Epoch: 49 | Time: 0:01:11.412961 | Train SE: 12.237 | Test SE: 12.355
Epoch: 50 | Time: 0:01:12.447602 | Train SE: 12.216 | Test SE: 12.451
Epoch: 51 | Time: 0:01:15.105566 | Train SE: 12.244 | Test SE: 12.327

<Figure size 640x480 with 1 Axes>
Epoch: 52 | Time: 0:01:13.765768 | Train SE: 12.125 | Test SE: 12.234
Epoch: 53 | Time: 0:01:11.060662 | Train SE: 12.222 | Test SE: 12.442
Epoch: 54 | Time: 0:01:11.167387 | Train SE: 12.263 | Test SE: 12.452

<Figure size 640x480 with 1 Axes>
Epoch: 55 | Time: 0:01:11.714297 | Train SE: 12.291 | Test SE: 12.504
Epoch: 56 | Time: 0:01:11.125837 | Train SE: 12.390 | Test SE: 12.401
Epoch: 57 | Time: 0:01:10.899073 | Train SE: 12.289 | Test SE: 12.406

<Figure size 640x480 with 1 Axes>
Epoch: 58 | Time: 0:01:15.732272 | Train SE: 12.354 | Test SE: 12.467
Epoch: 59 | Time: 0:01:14.940771 | Train SE: 12.293 | Test SE: 12.350
Epoch: 60 | Time: 0:01:14.773357 | Train SE: 12.284 | Test SE: 12.449

<Figure size 640x480 with 1 Axes>
Epoch: 61 | Time: 0:01:15.458194 | Train SE: 12.339 | Test SE: 12.295
Epoch: 62 | Time: 0:01:15.566866 | Train SE: 12.217 | Test SE: 12.291
Epoch: 63 | Time: 0:01:15.840924 | Train SE: 12.253 | Test SE: 12.503

<Figure size 640x480 with 1 Axes>
Epoch: 64 | Time: 0:01:16.002238 | Train SE: 12.362 | Test SE: 12.561
Epoch: 65 | Time: 0:01:17.569300 | Train SE: 12.332 | Test SE: 12.360
Epoch: 66 | Time: 0:01:15.864428 | Train SE: 12.278 | Test SE: 12.451

<Figure size 640x480 with 1 Axes>
Epoch: 67 | Time: 0:01:15.666432 | Train SE: 12.387 | Test SE: 12.551
Epoch: 68 | Time: 0:01:15.576704 | Train SE: 12.402 | Test SE: 12.383
Epoch: 69 | Time: 0:01:16.139608 | Train SE: 12.334 | Test SE: 12.527

<Figure size 640x480 with 1 Axes>
Epoch: 70 | Time: 0:01:16.381742 | Train SE: 12.420 | Test SE: 12.592
Epoch: 71 | Time: 0:01:16.760356 | Train SE: 12.459 | Test SE: 12.474
Epoch: 72 | Time: 0:01:17.043829 | Train SE: 12.398 | Test SE: 12.527

<Figure size 640x480 with 1 Axes>
Epoch: 73 | Time: 0:01:16.008651 | Train SE: 12.407 | Test SE: 12.468
Epoch: 74 | Time: 0:01:15.891396 | Train SE: 12.386 | Test SE: 12.401
Epoch: 75 | Time: 0:01:16.242761 | Train SE: 12.271 | Test SE: 12.515
```
