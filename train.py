# %%
import torch
import os
import random
import time
import numpy as np
import torch.utils.data as Data
import datetime
import Network_FDD_CBAM as FDD
import logging
import sys

# --- 全局环境变量设置 ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- 确保目录存在 ---
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
if not os.path.exists('logs'):
    os.makedirs('logs')
DATA_DIR = "data"

EXPECTED_SE_MAP = {
    1: 4.0,
    3: 6.0,
    16: 9.0,
    24: 10.0,
    32: 11.0,
    48: 12.0,
    64: 12.0,
}

# --- 随机种子设置函数 ---
def set_seed(seed, logger):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"已设置种子为: {seed}")


# --- 日志设置函数 ---
def setup_logger(log_name):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_filename = os.path.join("logs", f"{log_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__), log_filename


def checkConvergence(epoch,
                     test_losses,
                     pass_score,  # 从外部传入的预期值
                     start_epoch=20,
                     window_size=15,
                     growth_threshold=0.1):
    """
    检查训练是否提前收敛的函数。
    只有当【性能增长停滞】且【未达到预期值】时，才会触发。
    """
    # 在初期阶段不进行检查
    if epoch < start_epoch:
        return False

    # 如果当前性能已经超过了预期值(pass_score)，我们认为这是一个“好的”收敛，不提前终止。
    # 这样可以确保模型有机会在高性能区域继续微调。
    if test_losses[-1] > pass_score:
        return False

    # 检查最近window_size个epoch的性能增长是否都低于阈值
    if len(test_losses) < window_size + 1:
        return False

    recent_performance = test_losses[-(window_size + 1):]
    for i in range(window_size):
        growth = recent_performance[i + 1] - recent_performance[i]
        if growth >= growth_threshold:
            # 只要有一次增长大于阈值，就认为没有停滞
            return False

    # 如果连续 window_size 轮增长都低于阈值，并且性能还没达标，则判断为收敛
    logging.getLogger(__name__).info(f"\n[Convergence Watcher] 在第 {epoch} 轮检测到性能停滞!")
    logging.getLogger(__name__).info(f"    - 当前性能 {test_losses[-1]:.4f} 未达到预期值 {pass_score:.4f}")
    logging.getLogger(__name__).info(f"    - 且已连续 {window_size} 轮的性能增长低于阈值 ({growth_threshold})")
    logging.getLogger(__name__).info("    - 提前终止本次训练。")
    return True

# %%
def train(Nc, N, Nt, B, Nr, L, SNR_dB, K, EPOCH, BATCH_SIZE):
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    param_str = f"B{B}_N{N}_L{L}_K{K}_SNR{SNR_dB}"
    log_name = f"{run_id}_{param_str}"
    model_savename_base = f"{run_id}_{param_str}.pth"
    logger, _ = setup_logger(log_name)

    logger.info("=" * 30 + " 开始新的训练任务 " + "=" * 30)
    logger.info(f"参数: B={B}, N={N}, L={L}, K={K}, SNR_dB={SNR_dB}")
    logger.info(f"运行ID: {run_id}")

    pass_score = EXPECTED_SE_MAP.get(B, 0)
    logger.info(f"本次运行的性能预期值 (pass_score): {pass_score}")

    seed = int(time.time() * 1000) % (2 ** 32 - 1)
    set_seed(seed, logger)

    shoulian = np.zeros(EPOCH)
    test_losses_history = []
    snr = 10 ** (SNR_dB / 10) / K
    parm_set = [Nc, Nt, Nr, snr, B, K]

    logger.info("正在加载数据...")
    H_train_1 = torch.load(f'data/H_train_UPA{N}Lp_1.pt')[:, 0:K, :, :]
    H_train_2 = torch.load(f'data/H_train_UPA{N}Lp_2.pt')[:, 0:K, :, :]
    H_train = torch.cat([H_train_1, H_train_2], 0)
    H_test = torch.load(f'data/H_test_UPA{N}Lp.pt')[:, 0:K, :, :]
    logger.info(f"数据加载成功。")

    net_US = FDD.DNN_US_RF_OFDM(parm_set).cuda()
    net_BS = FDD.DNN_BS_hyb_OFDM(parm_set).cuda()

    optimizer_US = torch.optim.Adam(net_US.parameters(), lr=0.001)
    scheduler_US = torch.optim.lr_scheduler.MultiStepLR(optimizer_US, milestones=[100, 150], gamma=0.6)
    optimizer_BS = torch.optim.Adam(net_BS.parameters(), lr=0.001)
    scheduler_BS = torch.optim.lr_scheduler.MultiStepLR(optimizer_BS, milestones=[100, 150], gamma=0.6)

    loss_func1 = FDD.MyLoss_OFDM().cuda()

    loader_train = Data.DataLoader(Data.TensorDataset(H_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    loader_test = Data.DataLoader(Data.TensorDataset(H_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_test_se = -np.inf
    converged = False

    for epoch in range(EPOCH):
        train_SE, num_train = 0, 0
        test_SE, num_test = 0, 0

        net_US.train()
        net_BS.train()
        for step, [b_x] in enumerate(loader_train):
            num_train += 1
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

        logger.info(f'Epoch: {epoch:03d}/{EPOCH - 1} | Train SE: {train_SE:.4f} | Test SE: {test_SE:.4f}')

        shoulian[epoch] = test_SE
        test_losses_history.append(test_SE)

        if test_SE > best_test_se:
            best_test_se = test_SE
            torch.save(net_US, os.path.join('saved_models', f'net_US_{model_savename_base}'))
            torch.save(net_BS, os.path.join('saved_models', f'net_BS_{model_savename_base}'))
            logger.info(f'---> New best model saved with Test SE: {best_test_se:.4f}')

        if checkConvergence(epoch, test_losses_history, pass_score):
            converged = True
            break

    logger.info("\n" + "=" * 30 + " 训练结束 " + "=" * 30)
    logger.info(f"最佳测试集 SE: {best_test_se:.4f}")
    return best_test_se, converged


if __name__ == '__main__':
    Nc = 32
    Nt = 64
    Nr = 1
    L = 4
    N = 2
    SNR_dB = 10
    K = 2
    BATCH_SIZE = 512
    EPOCH = 180

    B_values = [1, 3, 16, 24, 32, 48, 64]
    results_summary = []

    # --- 新的执行逻辑：对每个B值运行一次，并保存所有结果 ---
    for B in B_values:
        print("\n" + "#" * 80)
        print(f"######  开始处理参数 B = {B}  ######")
        print("#" * 80 + "\n")

        best_se, did_converge = train(Nc, N, Nt, B, Nr, L, SNR_dB, K, EPOCH, BATCH_SIZE)

        status = "提前收敛" if did_converge else "完成所有Epoch"
        summary = f"参数 B={B}: 最佳SE = {best_se:.4f}, 状态 = {status}"
        results_summary.append(summary)

        print("\n" + "-" * 80)
        print(f"######  B = {B} 的训练已结束。结果已保存。  ######")
        print(f"######  总结: {summary}  ######")
        print("-" * 80 + "\n")

    print("\n\n" + "=" * 30 + " 所有实验总结 " + "=" * 30)
    for res in results_summary:
        print(res)
    print("=" * 75)