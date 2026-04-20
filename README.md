# 作业：基于深度学习的心电数据分类

## 项目概述

使用 MIT-BIH 心电图数据集，构建并训练深度学习模型对 ECG 信号进行 5 类分类（N/S/V/F/Q），与机器学习方法进行统计比较，并系统性地探究模型设计选择。

## 数据集

- **来源**: [Heartbeat Dataset (Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data)
- **文件**: `data/raw/mitbih_train.csv`, `data/raw/mitbih_test.csv`
- **类别**: 5 类 — N(0), S(1), V(2), F(3), Q(4)
- **采样率**: 125Hz，每条样本裁剪/重采样至 188 维（最后一列为标签）

## 统一训练配置

所有 Phase 共享以下 baseline（仅被探究的变量改变）：

- **模型**: ResNet1D(kernel_size=7, use_se=False)
- **损失**: FocalLoss(alpha=class_weights, gamma=2.0)
- **优化**: AdamW, lr=1e-3, weight_decay=1e-4
- **训练**: epochs=100, early stopping(patience=15), ReduceLROnPlateau
- **数据**: train_df 分出 10% 验证集（分层抽样），test_df 仅用于最终评估
- **统计**: N_TRIALS=5（种子 42-46），定量结论配合统计检验

## 实验阶段

| Phase | 内容 | 统计方法 |
|-------|------|---------|
| 1 | 1D-ResNet 基线 + ML/DL 统计对比 | 配对 T 检验 |
| 2 | 感受野(Kernel Size) + 超参网格搜索 | ANOVA + Bonferroni |
| 3.1 | 输入模态对比 (Raw/FFT/Mel/CWT) | ANOVA + Bonferroni |
| 3.2 | 架构归纳偏置 (ResNet/BiLSTM/MLP-Mixer) | ANOVA + Bonferroni |
| 3.3 | 分类器解耦 (Softmax/RF/SVM) | ANOVA + Bonferroni |
| 3.4 | 数据增强鲁棒性 | F1 衰减率对比 |
| 3.5 | R 波对齐漂移 (CNN vs MLP-Mixer vs RF) | F1 衰减率对比 |
| 4 | SE 注意力 + Grad-CAM 可解释性 | 配对 T 检验 |
| 5 | InceptionTime vs ResNet1D+SE | 配对 T 检验 |

## 运行顺序

Phase 1 必须最先运行，其余 Phase 依赖关系：

```
Phase 1 → Phase 2 (K=7), Phase 3.1 (1D_Raw), Phase 3.2 (1D-ResNet),
          Phase 3.3, Phase 3.4 (baseline), Phase 3.5 (CNN), Phase 4 (SE=False)
Phase 3.2 → Phase 3.5 (MLP-Mixer)
```

```bash
# 依次运行
python scripts/run_phase1.py
python scripts/run_phase2.py
python scripts/run_phase3_1.py          # 加 --resume 跳过已有 checkpoint
python scripts/run_phase3_2.py
python scripts/run_phase3_3.py
python scripts/run_phase3_4.py
python scripts/run_phase3_5.py
python scripts/run_phase4.py
python scripts/run_phase5.py
```

## 项目结构

```
data/
  raw/                          原始数据 (mitbih_train.csv, mitbih_test.csv)
  processed/                    处理后数据
scripts/                        可执行脚本
  run_phase1.py ~ run_phase5.py
src/                            源代码包
  data/
    loader.py                   数据加载与 DataLoader 构造
    dataset.py                  ECGDataset (支持 raw/fft/mel/cwt、augment、jitter)
  features/
    transforms.py               模态转换 (FFT/Mel/CWT) 与数据增强 (jitter/drift/noise)
  models/
    resnet.py                   SEBlock1D, BasicBlock1D, ResNet1D, ResNet2D
    inception.py                InceptionModule, InceptionBlock, InceptionTime
    sequence.py                 BiLSTM, MixerLayer, MLPMixer
  training/
    trainer.py                  ECGTrainer (fit/evaluate, early stopping, LR scheduler)
    losses.py                   FocalLoss, compute_class_weights
  evaluation/
    embeddings.py               get_embeddings (Phase 3.3)
    interpretability.py         GradCAM1D (Phase 4)
  utils/
    tools.py                    set_seed, get_device, count_parameters, measure_latency
    logger.py                   save_json
results/
  logs/                         实验结果 JSON
  models/                       模型权重 (.pth)
  figures/                      可视化图表 (.png)
notebooks/
  figures.ipynb                 实验结果可视化 (生成 results/figures/ 下所有图表)
```
