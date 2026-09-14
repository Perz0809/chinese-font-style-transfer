#!/bin/bash
#SBATCH --job-name=font-gan-l2
#SBATCH --partition=gpu-t4
#SBATCH --gres=gpu:1          
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=pipeline_task5.log

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=== 1. 专门补跑 Task5_GAN_L2 ==="
$PYTHON_BIN -u train.py --gpu 0 --tasks Task5_GAN_L2 --mode train > train_task5_only.log 2>&1

echo "=== 2. 执行统一评估 ==="
$PYTHON_BIN -u train.py --mode eval > eval.log 2>&1

echo "=== 3. 执行推理 (Inference) ==="
$PYTHON_BIN -u inference.py > inference.log 2>&1

echo "=== 4. 生成学术图表 (Plotting) ==="
$PYTHON_BIN -u plot_curves.py > plot.log 2>&1

echo "=== 补跑流水线全部完成: $(date) ==="
