#%%
import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import importlib
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
# %matplotlib notebook
from matplotlib import cm
from scipy.linalg import block_diag
import datetime
from torch.nn.utils import *
from torch.utils.data import DataLoader, TensorDataset
parser = argparse.ArgumentParser(description='自动化训练不同版本的模型')
parser.add_argument('--model_version', type=str, required=True, choices=['original', 'conformer'],
                    help='要运行的模型版本: "original" 或 "conformer"')
parser.add_argument('--epochs', type=int, default=180, help='训练的总轮次')
args = parser.parse_args()

# 2. 根据命令行传入的参数，动态加载对应的模型模块
print(f"--- 正在加载模型版本: {args.model_version.upper()} ---")
if args.model_version == 'original':
    FDD = importlib.import_module('Network_FDD')
else:  # 'conformer'
    FDD = importlib.import_module('Network_FDD_Conformer')

#%%
# 创建用于存放模型和结果图的文件夹
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
if not os.path.exists('results'):
    os.makedirs('results')
#%%
def train(Nc, N, Nt, B, Nr, L, SNR_dB, K, EPOCH, BATCH_SIZE):
    shoulian = np.zeros(EPOCH)
    snr = 10**(SNR_dB/10) / K
    parm_set = [Nc, Nt, Nr, snr, B, K]

    H_train_1 = torch.load('data/H_train_UPA' + str(N) + 'Lp_1.pt')[:, 0:K, :, :]
    H_train_2 = torch.load('data/H_train_UPA' + str(N) + 'Lp_2.pt')[:, 0:K, :, :]
    H_train = torch.cat([H_train_1, H_train_2], 0)
    H_test = torch.load('data/H_test_UPA' + str(N) + 'Lp.pt')[:, 0:K, :, :]

    net_US = FDD.DNN_US_RF_OFDM(parm_set).cuda()
    net_BS = FDD.DNN_BS_hyb_OFDM(parm_set).cuda()


    optimizer_US = torch.optim.Adam(net_US.parameters(), lr=0.001)
    scheduler_US = torch.optim.lr_scheduler.MultiStepLR(optimizer_US, milestones=[100, 150], gamma=0.6)
    optimizer_BS = torch.optim.Adam(net_BS.parameters(), lr=0.001)
    scheduler_BS = torch.optim.lr_scheduler.MultiStepLR(optimizer_BS, milestones=[100, 150], gamma=0.6)
    loss_func1 = FDD.MyLoss_OFDM().cuda()

    loader_train = Data.DataLoader(Data.TensorDataset(H_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    loader_test = Data.DataLoader(Data.TensorDataset(H_test), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    train_losses, test_losses = [], []
    best_test_se = 0
    plt.ion()  # 开启交互模式

    start = datetime.datetime.now()
    for epoch in range(EPOCH):
        train_SE, num_train = 0, 0
        test_SE, num_test = 0, 0

        for step, [b_x] in enumerate(loader_train):
            num_train += 1
            net_US.train()
            net_BS.train()
            b_x = b_x.cuda()
            num = b_x.shape[0]
            out1 = torch.zeros([num, B * K]).cuda()
            for i in range(K):
                out1[:, i * B:(i * B + B)] = net_US(b_x[:, i, :, :], parm_set)
            out2 = net_BS(out1, parm_set)
            loss = loss_func1(b_x, out2, parm_set)
            train_SE -= loss.item()

            optimizer_US.zero_grad()
            optimizer_BS.zero_grad()
            loss.backward()
            optimizer_US.step()
            optimizer_BS.step()

        train_SE /= num_train
        scheduler_US.step()
        scheduler_BS.step()

        net_US.eval()
        net_BS.eval()
        with torch.no_grad():
            for step, [b_x] in enumerate(loader_test):
                num_test += 1
                b_x = b_x.cuda()
                num = b_x.shape[0]
                out1 = torch.zeros([num, B * K]).cuda()
                for i in range(K):
                    out1[:, i * B:(i * B + B)] = net_US(b_x[:, i, :, :], parm_set)
                out2 = net_BS(out1, parm_set)
                loss = loss_func1(b_x, out2, parm_set)
                test_SE -= loss.item()

        test_SE /= num_test
        time_elapsed = datetime.datetime.now() - start
        print(f'Epoch: {epoch} | Time: {time_elapsed} | Train SE: {train_SE:.3f} | Test SE: {test_SE:.3f}')
        start = datetime.datetime.now()

        train_losses.append(train_SE)
        test_losses.append(test_SE)

        if epoch > 0 and epoch % 10 == 0:
            plt.clf()
            plt.plot(train_losses, label='Train SE')
            plt.plot(test_losses, label='Test SE')
            plt.xlabel('Epoch')
            plt.ylabel('Spectral Efficiency')
            plt.title(f'Training Progress (B={B}, N={N}) - Epoch {epoch}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'results/progress_snapshot_{args.model_version}_{B}B{N}Lp{L}L{K}K_epoch{epoch}.png')
            plt.pause(0.1)

        if test_SE > best_test_se:
            best_test_se = test_SE
            torch.save(net_US, f'saved_models/net_US_{args.model_version}_{B}B{N}Lp{L}L{K}K.pth')
            torch.save(net_BS, f'saved_models/net_BS_{args.model_version}_{B}B{N}Lp{L}L{K}K.pth')
            print(f'New best model saved with Test SE: {best_test_se:.4f}')

        shoulian[epoch] = test_SE

    plt.ioff()
    plt.clf()
    plt.plot(train_losses, label='Train SE')
    plt.plot(test_losses, label='Test SE')
    plt.axhline(y=best_test_se, color='r', linestyle='--', label=f'Best Test SE: {best_test_se:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Spectral Efficiency')
    plt.title(f'Final Training Progress (B={B}, N={N})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/final_progress_{args.model_version}_{B}B{N}Lp{L}L{K}K.png')
    plt.show()
    print(f'The best SE is: {max(test_losses):.3f}')
    print(shoulian)
#%%

Nc = 32 #number of subcarriers
N = 2   # Number of paths
Nt = 64 # Number of Antennas at the BS
Nr = 1  # Number of Antennas at the UE
# B = 30

L = 8   # number of pilot OFDM symbols
SNR_dB = 10  # SNR
K = 2   # number of UEs
snr =  10**(SNR_dB/10)/K

BATCH_SIZE = 512
# EPOCH = 180

for B in [64]:
    train(Nc, N, Nt, B, Nr, L, SNR_dB, K, args.epochs, BATCH_SIZE)