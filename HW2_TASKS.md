# 作业二（基于深度学习的心电数据分类）

## 数据集
- 来源：Heartbeat Dataset
  - https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data
- 数据：
  1. 文件 `data/raw/mitbih_train.csv`、`data/raw/mitbih_test.csv`
  2. Number of Categories: 5
  3. Sampling Frequency: 125Hz
  4. Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
  5. This dataset consists of a series of CSV files. Each of these CSV files contain a matrix, with each row representing an example in that portion of the dataset. The final element of each row denotes the class to which that example belongs.
  6. All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 188.


---

## 作业总体目标
- 构造并训练深度学习模型对心电（ECG）信号进行分类。与作业一机器学习结果进行比较；讨论模型设计与超参数选择原则。

---

## 统一训练配置

所有 Phase 共享以下 baseline 配置（仅被探究的变量改变）：

- **模型**：ResNet1D(kernel_size=7, use_se=False)（标准 ResNet-18 深度，4-stage BasicBlock）
- **损失**：FocalLoss(alpha=class_weights, gamma=2.0)
- **优化**：AdamW, lr=1e-3, weight_decay=1e-4
- **训练**：epochs=100, early stopping(patience=15), ReduceLROnPlateau
- **数据分割**：`create_dataloaders(train_df, test_df)` 从 train_df 分出 10% 验证集（分层抽样），test_df 仅用于最终评估，训练过程从未见过
- **统计**：N_TRIALS=5 独立试验（种子 42-46），定量结论均配合统计检验

---

## 深度学习心电分类进阶预估任务清单

### Phase 1: 1D-ResNet 基线构建与严谨的 ML/DL 统计对比

- **任务 1.1**：构建带残差连接的 1D-ResNet 基线（ResNet-18 深度），引入 Focal Loss。
   - 应对不平衡：在损失函数中引入类别反比权重（sklearn balanced 风格）。
- **任务 1.2**：跑 `N_TRIALS=5` 次独立试验（不同随机种子），ML 端用 RF + SMOTE，DL 端用 1D-ResNet + FocalLoss，统一在固定测试集上评估 F1-macro，对两组测试集分数进行**配对 T 检验** (`p < 0.05`)。
- **保存**：`results/logs/phase1_stats.json`（ML/DL 各 trial F1、训练历史、混淆矩阵、每类 F1、t/p 值）
- **模型权重**：`results/models/Phase1_ResNet_T{0-4}/best_model.pth`

### Phase 2: 模型架构与超参数的动力学探究

- **任务 2.1**：感受野 (Kernel Size) 与网络深度的耦合效应（对比 Train/Val Loss 收敛状态）。
   - 对比 kernel_size = 3, 7, 15, 31。其中 K=7 复用 Phase 1 结果。
   - **统计要求**：`N_TRIALS=5` 次独立试验，测试集 F1-macro，使用单因素 **ANOVA + Bonferroni 校正** 事后配对 T 检验。
- **任务 2.2**：优化空间的平滑性探讨（网格搜索 LR × Weight Decay，绘制热力图）。单次运行，属于探索性实验。
- **保存**：`phase2_kernel_dynamics.json`, `phase2_hyper_grid.json`

### Phase 3: 深度学习全链路解耦消融实验 (核心控制变量区)

> 将整个流水线拆分为三个模块，每次只改变其中一个模块，严格控制其他变量。

#### 实验 3.1：输入特征的模态之争
*(Fix Extractor & Classifier, Change Input)*

**控制变量**：1D 使用 ResNet1D，2D 使用 ResNet2D。1D_Raw 结果复用 Phase 1。
- **1D 对比**：原始时间序列 (Raw) vs. 频域序列 (FFT)。
- **2D 对比**：梅尔频谱图 (Mel-Spectrogram) vs. 连续小波变换图 (CWT)。
- **统计要求**：每种模态 `N_TRIALS=5` 次，**ANOVA + Bonferroni 事后检验**。
- **支持 `--resume`**：跳过已有 checkpoint 的 trial，直接加载评估。

#### 实验 3.2：特征提取器的归纳偏置之争
*(Fix Input & Classifier, Change Extractor)*

**控制变量**：固定输入为 1D 原始时间序列。1D-ResNet 结果复用 Phase 1。
- **对比对象**：1D-ResNet-18（局部感受野） vs. BiLSTM（全局序列记忆） vs. MLP-Mixer（纯 MLP token/channel 交替）。
- **公平性**：三个架构均使用文献标准配置（ResNet-18 ~8.7M, BiLSTM h=256/L=3 ~3.7M, MLP-Mixer d=256/L=8 ~4.3M）。
- **统计要求**：`N_TRIALS=5` 次，**ANOVA + Bonferroni 事后检验**。评估维度：F1-macro（均值±标准差）、参数量、推理延迟。

#### 实验 3.3：分类决策边界之争
*(Fix Input & Extractor, Change Classifier)*

**控制变量**：加载 Phase 1 已训练的 1D-ResNet 模型（冻结权重），提取倒数第二层 Embedding。
- **对比对象**：End-to-End Softmax vs. Random Forest (RF) vs. SVM (RBF)。
- **统计要求**：`N_TRIALS=5` 次（加载 Phase 1 的 5 个模型），**ANOVA + Bonferroni 事后检验**。

#### 实验 3.4：领域特定的数据增强
*(Domain-Specific Augmentation)*

- 引入医学现实中常见的噪声（0.5Hz 基线漂移、高斯白噪声），对比无增强和有增强模型在带噪测试集上的 F1 衰减率。
- Baseline（无增强）复用 Phase 1 trial 0 结果，augmented 模型独立训练。

#### 实验 3.5：临床现实场景模拟——可穿戴设备 R 波对齐漂移测试
*(R-Peak Misalignment Robustness)*

- **实验设计**：在测试集上引入随机均匀抖动（jitter = 0, 2, 5, 10, 15 步）。
- **对比对象**：CNN（ResNet1D，含池化层） vs. MLP-Mixer（无池化/卷积） vs. RF（ML baseline）。
  - CNN 和 MLP-Mixer 加载 Phase 1 / Phase 3.2 已训练模型，无需重新训练。
  - RF 和 DL 均使用预计算的抖动特征，非零 jitter 做 5 次随机抖动取平均。
- **验证假设**：CNN 的池化层带来平移鲁棒性，MLP-Mixer 和 RF 对时序抖动更敏感。

### Phase 4: 黑盒破局：注意力机制与可解释性

- **任务 4.1**：在 ResNet 中嵌入 SE (Squeeze-and-Excitation) 通道注意力模块，对比 SE=True vs SE=False。SE=False baseline 复用 Phase 1 结果。
- **任务 4.2**：结合 Grad-CAM，针对各类异常心跳（S/V/F/Q），生成《心电病理特征解释图册》（波形 + 注意力热力图）。
- **保存**：`phase4_se_gradcam.json`（SE/No-SE F1 对比、训练历史、病理图册）

### Phase 5: SOTA 模型引入与复现

- **架构**：InceptionTime (Fawaz et al., 2019) — 6 个 InceptionBlock 堆叠，每层 bottleneck + 并行多尺度卷积(K=3/5/7) + MaxPool + residual shortcut。
- **对比**：InceptionTime vs. ResNet1D(kernel_size=7, use_se=True)，`N_TRIALS=5` 次，**配对 T 检验** (`p < 0.05`)。

### Phase 6: 规范化的工程提交

- 模块化代码（`src/data/`, `src/models/`, `src/training/` 等）。
- 所有实验结果保存为 JSON（`results/logs/`），支持下游直接读取绘图。
- 模型权重保存为 `.pth`（`results/models/`），支持跨 Phase 复用。

---

## 运行顺序

Phase 1 必须最先运行（后续 Phase 依赖其模型权重和 JSON 结果），其余 Phase 依赖关系：

```
Phase 1 → Phase 2 (K=7), Phase 3.1 (1D_Raw), Phase 3.2 (1D-ResNet),
          Phase 3.3, Phase 3.4 (baseline), Phase 3.5 (CNN), Phase 4 (SE=False)
Phase 3.2 → Phase 3.5 (MLP-Mixer)
```

---

## 仓库结构
```
data/
  raw/                  原始数据 (mitbih_train.csv, mitbih_test.csv)
  processed/            处理后数据
scripts/                可执行脚本
  run_phase1.py ~ run_phase5.py
src/                    源代码包
  data/
    loader.py           数据加载 (load_train_test, create_dataloaders, make_loader)
    dataset.py          ECGDataset（支持 raw/fft/mel/cwt、augment、jitter）
  features/
    transforms.py       模态转换 (FFT/Mel/CWT) 与数据增强 (jitter/drift/noise)
  models/
    resnet.py           SEBlock1D, BasicBlock1D, ResNet1D, BasicBlock2D, ResNet2D
    inception.py        InceptionModule, InceptionBlock, InceptionTime
    sequence.py         BiLSTM, MixerLayer, MLPMixer
  training/
    trainer.py          ECGTrainer (fit/evaluate, early stopping, LR scheduler)
    losses.py           FocalLoss, compute_class_weights
  evaluation/
    embeddings.py       get_embeddings
    interpretability.py GradCAM1D
  utils/
    tools.py            set_seed, get_device, count_parameters, measure_latency
    logger.py           save_json
results/
  logs/                 实验结果 JSON
  models/               模型权重 (.pth)
```
