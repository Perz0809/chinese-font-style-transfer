# WGAN-GP 与 VAE 生成式字体风格迁移

## 项目目标

围绕中文字体风格迁移，研究 U-Net、VAE 和 WGAN-GP 等生成模型在字形结构保持、笔画细节和风格还原方面的表现，并通过消融实验比较跳跃连接、损失函数和模型结构的影响。

## 实现内容

- 使用 `preprocess.py` 从源字体和 5 种目标字体生成训练样本，清理空白字形和缺失字形，构建平衡数据集。
- 使用 U-Net 作为基础生成器，比较带跳跃连接和不带跳跃连接的结构。
- 在基础模型上加入 VAE 潜变量分支和 WGAN-GP 判别器，分别使用 L1、L2 重建损失进行实验。
- 记录训练损失、GPU 状态和模型检查点，使用推理脚本生成唐诗字体效果图。
- 使用 L1 Error、MSE、SSIM、Ink IoU、边缘误差和笔画细节等指标评价生成结果。

## 代码说明

- `preprocess.py`：字体渲染、缺失字形过滤和训练数据打包。
- `dataset.py`：读取字体数据集。
- `models.py`：U-Net、VAE、判别器和 WGAN-GP 损失实现。
- `train.py`：执行基线、消融、VAE 和 GAN 训练任务。
- `inference.py`：加载检查点并生成测试字体图像。
- `plot_curves.py`：绘制训练曲线和对比图。
- `autorun_all.sh`、`run_task5.sh`：批量训练脚本。

## 结果

在保存的评估结果中，Task5 的 U-Net + Skip + GAN + L2 配置取得 SSIM 0.924、Ink IoU 0.818；Task3 的 U-Net + Skip + VAE + L1 配置取得 SSIM 0.922、Ink IoU 0.803。相关图表和指标文件位于 `analysis-plots`、`model_evaluation_metrics.csv` 和 `best_model_config.txt`。

## 运行方式

代码中的默认路径面向课程服务器。复现前请在 `preprocess.py`、`train.py` 和相关脚本中修改字体、数据集和模型权重路径。

```bash
python preprocess.py
python train.py
python inference.py
python plot_curves.py
```

## 说明

仓库保留了代码、实验图表和评估结果；字体文件、训练权重和 GPU 运行日志不作为作品集必需内容上传。
