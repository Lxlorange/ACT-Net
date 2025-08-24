# %%
import os

import os
import torch
import numpy as np
from math import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %%
def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(- 2 * pi * 1J / N)
    W = np.power(omega, i * j) / sqrt(N)
    return np.mat(W)


# %% md
# 均匀面阵
# %%
# 最终修正版函数
# 修正后的 channel_OFDM.ipynb 中的函数
def freqency_sparse_SV_channel0(Nc,N,sigma_2_alpha,Nt,Nr): # 添加 Nr 作为参数
    # Nc代表载波数 N代表路径数 sigma_2_alpha每条路径能量
    d=0.5
    Tau = Nc  # 最大路径时延
    tau = np.random.rand(1,N)*Tau

    # 将Nt从列表转换为整数，以供后续使用
    if isinstance(Nt, list):
        N_t = Nt[0]*Nt[1]
    else:
        N_t = Nt

    N_r = Nr # 明确接收天线数

    # 为UPA（均匀平面阵列）或ULA（均匀线性阵列）生成导向矢量
    fait = np.random.rand(1,N)*2*pi
    fair = np.random.rand(1,N)*2*pi
    theatt = np.random.rand(1,N)*2*pi
    theatr = np.random.rand(1,N)*2*pi

    A_t = np.mat(np.zeros([N_t,N],dtype = complex))
    A_r = np.mat(np.zeros([N_r,N],dtype = complex)) # <--- 修正点 1: A_r 的维度
    alpha = np.squeeze(np.zeros([1,N],dtype = complex),0)

    # ***** 关键修正 *****
    # H_f 的维度应为 [Nr, N_t, Nc]，为了方便后续按子载波索引，先定义为 [Nr, Nc, N_t]
    H_f = np.array(np.zeros([N_r,Nc,N_t],dtype = complex)) # <--- 修正点 2: 保证 H_f 形状正确

    # 生成每个路径的导向矢量和复数增益
    for i in range(N):
        # 为发射端(BS)生成导向矢量 (UPA)
        if isinstance(Nt, list):
            n_t1 = np.mat(range(Nt[0])).reshape(Nt[0],1)
            n_t2 = np.mat(range(Nt[1])).reshape(Nt[1],1)
            at1 = np.exp(-2j*pi*d*n_t1*np.cos(fait[0,i])*np.sin(theatt[0,i]))
            at2 = np.exp(-2j*pi*d*n_t2*np.sin(fait[0,i]))
            A_t[:,i] = np.kron(at1,at2)
        else: # ULA
            n_t = np.mat(range(N_t)).reshape(N_t,1)
            A_t[:,i] = np.exp(-2j*pi*d*n_t*np.sin(fait[0,i]))

        # 为接收端(UE)生成导向矢量 (假设为ULA)
        n_r = np.mat(range(N_r)).reshape(N_r,1)
        A_r[:,i] = np.exp(-2j*pi*d*n_r*np.sin(fair[0,i])) # <--- 修正点 3: 计算 A_r

        # 生成路径复数增益
        aa = (np.random.randn(1,1)+1j*np.random.randn(1,1))*np.sqrt(sigma_2_alpha/2/N)
        alpha[i] = aa[0,0]

    # 在频域上合成每个子载波的信道
    for k in range(Nc):
        P = np.squeeze(alpha*np.exp(-1j*2*pi*tau*k/Nc),0);
        P = np.diag(P)
        # H_f(k) = A_r * diag(alpha_k) * A_t^H
        H_f[:,k,:] = np.dot(np.dot(A_r,P),A_t.H) # <--- 修正点 4: 正确合成信道

    # 返回的 H_f 形状是 [Nr, Nc, N_t]
    return H_f, A_t, alpha

# %%
Nc = 16
sigma_2_alpha = 1
Nt = [12, 12]
N_t = Nt[0] * Nt[1]
Nr = 16

B = 30
D = 1000;  # 角度采样点数
L = 8

SNR_dB = 5
K = 4
snr = 10 ** (SNR_dB / 10) / K

N_BATCH_train = 40
N_BATCH_test = 8
BATCH_SIZE = 128
N_H_train = N_BATCH_train * BATCH_SIZE
N_H_test = N_BATCH_test * BATCH_SIZE

os.makedirs('data2', exist_ok=True)
# %%

for N in range(2, 3):
    print("N")
    H_torch = torch.zeros([BATCH_SIZE * N_BATCH_train, K, Nr, Nc, N_t * 2])
    for i in range(N_BATCH_train):
        print("i")
        H = np.zeros([BATCH_SIZE, K, Nr, Nc, N_t], dtype=complex)  # 第0个维度是样本 第1个维度是用户，第2个维度是子载波，第3个维度是天线
        for j in range(BATCH_SIZE):
            for k in range(K):
                print("k")
                H_f, A_t, alpha = freqency_sparse_SV_channel0(Nc, N, sigma_2_alpha, Nt, Nr)
                H[j, k, :, :, :] = H_f

        H_torch[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE), :, :, :, 0:N_t] = torch.from_numpy(np.real(H))
        H_torch[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE), :, :, :, N_t:2 * N_t] = torch.from_numpy(np.imag(H))
        print(i)
    torch.save(H_torch, 'data2/H_train_UPA' + str(N) + 'Lp_1.pt')

    H_torch = torch.zeros([BATCH_SIZE * N_BATCH_train, K, Nr, Nc, N_t * 2])
    for i in range(N_BATCH_train):
        H = np.zeros([BATCH_SIZE, K, Nr, Nc, N_t], dtype=complex)  # 第0个维度是样本 第1个维度是用户，第2个维度是子载波，第3个维度是天线
        for j in range(BATCH_SIZE):
            for k in range(K):
                # 在调用时传入 Nr
                H_f, A_t, alpha = freqency_sparse_SV_channel0(Nc, N, sigma_2_alpha, Nt, Nr)
                H[j, k, :, :, :] = H_f

        H_torch[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE), :, :, :, 0:N_t] = torch.from_numpy(np.real(H))
        H_torch[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE), :, :, :, N_t:2 * N_t] = torch.from_numpy(np.imag(H))
        print(i)
    torch.save(H_torch, 'data2/H_train_UPA' + str(N) + 'Lp_2.pt')
# %%
for N in range(2, 3):
    H_torch = torch.zeros([BATCH_SIZE * N_BATCH_test, K, Nr, Nc, N_t * 2])
    for i in range(N_BATCH_test):
        H = np.zeros([BATCH_SIZE, K, Nr, Nc, N_t], dtype=complex)  # 第0个维度是样本 第1个维度是用户，第2个维度是子载波，第3个维度是天线
        for j in range(BATCH_SIZE):
            for k in range(K):
                # 在调用时传入 Nr
                H_f, A_t, alpha = freqency_sparse_SV_channel0(Nc, N, sigma_2_alpha, Nt, Nr)
                H[j, k, :, :, :] = H_f

        H_torch[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE), :, :, :, 0:N_t] = torch.from_numpy(np.real(H))
        H_torch[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE), :, :, :, N_t:2 * N_t] = torch.from_numpy(np.imag(H))
        print(i)
    torch.save(H_torch, 'data2/H_test_UPA' + str(N) + 'Lp.pt')
# %%
# import scipy.io as io
#
# H = np.zeros([BATCH_SIZE,K,Nc,N_t],dtype=complex) #第0个维度是样本 第1个维度是用户，第2个维度是子载波，第3个维度是天线
# for j in range(BATCH_SIZE):
#     for k in range(K):
#         H_f,A_t,alpha = freqency_sparse_SV_channel0(Nc,N,sigma_2_alpha,Nt)
#         H[j,k,:,:] = H_f
# H_torch = torch.zeros([BATCH_SIZE,K,Nc,N_t*2])
# H_torch[:,:,:,0:N_t] = torch.from_numpy(np.real(H))
# H_torch[:,:,:,N_t:2*N_t] = torch.from_numpy(np.imag(H))
# #torch.save(H_torch,'data/H_test_UPA.pt')
# print(Nt)
#
#
# dataNew = os.path.join('data2', 'H_UPA.mat')
# io.savemat(dataNew, {'H_UPA': H})
# %%
