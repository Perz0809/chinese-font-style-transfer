#!/bin/bash
#SBATCH --job-name=font-style-transfer
#SBATCH --partition=gpu-t4
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=pipeline.log

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=== 开始自动化流程: $(date) ==="

# Step 1: Preprocess (只需要运行一次)
echo "[1/4] Running Preprocessing..."
$PYTHON_BIN -u preprocess.py > preprocess.txt 2>&1

echo "=== 启动双卡并行训练 (GPU 0 & GPU 1) ==="

# Step 2 & 3: 真正的双卡并行核心逻辑
# 我们将 5 个任务拆分给两张卡，每张卡现在都是独立的进程，会有自己的 tqdm 进度条
# 使用 & 后缀让它们同时在后台运行

# GPU 0 负责：Baseline, VAE, GAN_L2 (3个任务)
$PYTHON_BIN -u train.py --gpu 0 --tasks Task1_Baseline Task3_VAE Task5_GAN_L2 > train_gpu0.log 2>&1 &

# GPU 1 负责：Ablation, GAN_L1 (2个任务)
$PYTHON_BIN -u train.py --gpu 1 --tasks Task2_Ablation Task4_GAN_L1 > train_gpu1.log 2>&1 &

# 等待所有后台任务完成
wait

echo "=== 训练完成，正在进行最终评估与绘图 ==="

# Step 4 & 5: Inference 和 Plotting
# 评估和绘图逻辑一般较快，可以单卡运行，或者保留你原有的逻辑
$PYTHON_BIN -u train.py --mode eval > eval.log 2>&1
$PYTHON_BIN -u plot_curves.py > plot.txt 2>&1

echo "=== 全部完成: $(date) ==="
