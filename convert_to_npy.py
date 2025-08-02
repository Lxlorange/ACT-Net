import torch
import os

# --- 配置参数 ---
SOURCE_DATA_DIR = 'F:/dataset/data'
FINAL_DATA_DIR = 'data'
K_USERS = 2

print("--- 开始创建最终数据集 (修复版) ---")
os.makedirs(FINAL_DATA_DIR, exist_ok=True)

# 循环处理 N 从 1 到 8
# for n_value in range(1, 9):
n_value = 1
print(f"\n===== 正在处理 N = {n_value} 的数据集 =====")

# --- 处理训练数据 ---
final_train_path = os.path.join(FINAL_DATA_DIR, f'H_train_N{n_value}.pt')
train_path_1 = os.path.join(SOURCE_DATA_DIR, f'H_train_UPA{n_value}Lp_1.pt')
train_path_2 = os.path.join(SOURCE_DATA_DIR, f'H_train_UPA{n_value}Lp_2.pt')

if os.path.exists(final_train_path):
    print(f"文件 '{os.path.basename(final_train_path)}' 已存在，跳过。")
elif not os.path.exists(train_path_1) or not os.path.exists(train_path_2):
    print(f"警告：缺少 N={n_value} 的原始训练文件，跳过。")
else:
    # 使用逻辑最清晰的方式：先加载，再合并，最后切片
    print("加载第一个训练文件...")
    train_data_1 = torch.load(train_path_1)

    print("加载第二个训练文件...")
    train_data_2 = torch.load(train_path_2)

    print("合并两个训练文件...")
    full_train_data = torch.cat((train_data_1, train_data_2), dim=0)

    del train_data_1
    del train_data_2

    print(f"切片数据，只保留前 {K_USERS} 个用户...")
    final_train_data = full_train_data[:, 0:K_USERS, :, :]

    del full_train_data

    print(f"保存最终训练数据到: {final_train_path} (形状: {final_train_data.shape})")
    torch.save(final_train_data, final_train_path)
    print("训练数据处理完成！")

final_test_path = os.path.join(FINAL_DATA_DIR, f'H_test_N{n_value}.pt')
test_path = os.path.join(SOURCE_DATA_DIR, f'H_test_UPA{n_value}Lp.pt')

if os.path.exists(final_test_path):
    print(f"文件 '{os.path.basename(final_test_path)}' 已存在，跳过。")
elif not os.path.exists(test_path):
    print(f"警告：缺少 N={n_value} 的原始测试文件，跳过。")
else:
    test_data = torch.load(test_path)
    final_test_data = test_data[:, 0:K_USERS, :, :]
    print(f"保存最终测试数据到: {final_test_path} (形状: {final_test_data.shape})")
    torch.save(final_test_data, final_test_path)

print("\n--- 所有数据处理完毕！---")