@echo OFF
ECHO.
ECHO #######################################################
ECHO ###   开始执行自动化训练流水线                      ###
ECHO #######################################################
ECHO.

ECHO [任务 1/2] 正在启动: 训练 ORIGINAL 模型 (180 轮)...
python train_main.py --model_version original --epochs 180

ECHO.
ECHO [任务 1/2] ORIGINAL 模型训练完成!
ECHO.
ECHO #######################################################
ECHO.

ECHO [任务 2/2] 正在启动: 训练 CONFORMER 模型 (180 轮)...
python train_main.py --model_version conformer --epochs 180

ECHO.
ECHO [任务 2/2] CONFORMER 模型训练完成!
ECHO.
ECHO #######################################################
ECHO ###              所有任务已成功执行完毕             ###
ECHO #######################################################
ECHO.

PAUSE