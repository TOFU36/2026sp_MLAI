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

## 深度学习心电分类进阶预估任务清单

### Phase 1: 1D-ResNet 基线构建与严谨的 ML/DL 统计对比

- **任务 1.1**：构建带残差连接的 1D-ResNet 基线，引入 Focal Loss。
   - 应对不平衡：继承作业一的经验，不盲目追求 Accuracy，在损失函数中引入 Weighted CrossEntropy（结合各类别的反比权重）。
- **任务 1.2**：使用完全相同的 5 折交叉验证数据划分，控制 DL 的随机种子。引入 F1-macro 分数，对 DL 和 ML (RF+SMOTE) 进行配对 T 检验 (`p < 0.05`)。

### Phase 2: 模型架构与超参数的动力学探究

- **任务 2.1**：感受野 (Kernel Size) 与网络深度的耦合效应（对比 Train/Val Loss 收敛状态）。
   - 假设：小卷积核（如 k=3）只能捕捉局部高频噪音或极微小的尖峰，大卷积核（如 k=15 或 k=31）才能覆盖完整的 QRS 波群并捕捉低频形态差异。
- **任务 2.2**：优化空间的平滑性探讨（网格搜索 LR 与 Weight Decay，绘制热力图）。

### Phase 3: 深度学习全链路解耦消融实验 (核心控制变量区)

> 将整个流水线拆分为三个模块，每次只改变其中一个模块，严格控制其他变量。

#### 实验 3.1：输入特征的模态之争
*(Fix Extractor & Classifier, Change Input)*

**控制变量**：同维度基线网络（1D 固定使用 1D-CNN；2D 固定使用 2D-CNN）。
- **1D 对比**：原始时间序列 (Raw Time-Series) vs. 频域序列 (FFT 幅值)。探究对于 1D-CNN，究竟是直接看波形好，还是看频率好。
- **2D 对比**：梅尔频谱图 (Mel-Spectrogram) vs. 连续小波变换图 (CWT)。探究在 2D-CNN 下，哪种时频特征更能凸显异常心搏的高频突变。
- **1D 与 2D 对比** 原始时间序列 (Raw Time-Series) vs. 频域序列 (FFT 幅值) vs. 梅尔频谱图 (Mel-Spectrogram) vs. 连续小波变换图 (CWT)。探究对于不同的特征，同样使用 1D-CNN 在时序维度处理，判断不同特征的不同表现。

#### 实验 3.2：特征提取器的归纳偏置之争
*(Fix Input & Classifier, Change Extractor)*

**控制变量**：固定输入为 1D 原始时间序列。
- **对比对象**：1D-CNN（具备局部感受野） vs. Bi-LSTM（具备全局序列记忆） vs. MLP-Mixer（纯多层感知机交替）。
- **对比1D 与 2DExtractor**：利用实验3.1的数据，对比同样使用梅尔频谱图 (Mel-Spectrogram) 或 连续小波变换图 (CWT)特征，1D-CNN 与 2D-CNN 的结果有什么不同。（2 * 2 的组合方式，可能需要使用 ANOVA 判断是否有统计学的不同）
- **评估维度**：F1-macro、AUPRC、参数量 (Params) 和 推理延迟 (ms)。探讨哪种归纳偏置（Inductive Bias）最适合心电这一特定生理信号。

#### 实验 3.3：分类决策边界之争
*(Fix Input & Extractor, Change Classifier)*

**控制变量**：固定输入为 1D 原始时序，固定特征提取器为 **预训练并冻结权重** 的 1D-ResNet。
- **对比对象**：提取 ResNet 倒数第二层的 Embedding（表征向量），分别接入：End-to-End Softmax、Random Forest (RF)、支持向量机 (SVM)。
- **探究目的**：验证深度学习是否只是“特征提取得好”，而传统的 ML 分类器在处理高维稀疏 Embedding 时是否比 Softmax 更鲁棒。

#### 实验 3.4：领域特定的数据增强
*(Domain-Specific Augmentation)*

引入医学现实中常见的噪声（如模拟呼吸引起的 0.5Hz 基线漂移、传感器的高斯白噪声），对比无增强和有增强模型在带噪测试集上的衰减率。

#### ⭐ 实验 3.5：临床现实场景模拟——可穿戴设备 R 波对齐漂移测试
*(R-Peak Misalignment Test)*

- **临床背景**：便携心电贴在实时截取心电信号时，无法像 MIT-BIH 实验室数据那样将 QRS 波完美对齐在时间轴中央。
- **实验设计**：不再是简单的向右平移，而是在测试集上引入**随机均匀抖动 (Random Uniform Jitter)**，即让所有测试样本的波峰在 `[−10, +10]` 个时间步内随机左右偏移。对比 1D-CNN 与 RF 的鲁棒性差异。
- **临床级可视化**：推测 CNN 拥有池化层，对随机抖动有鲁棒性。提取 1D-CNN 在“不同抖动偏移量”下的 Grad-CAM 注意力热力图。证明 1D-CNN 拥有类似医生的“视线追踪”能力——不管 QRS 波平移到左边还是右边，红色的高亮注意力斑块始终“死死咬住”病变波形。

### Phase 4: 黑盒破局：注意力机制与可解释性

- **任务 4.1**：在最优模型中嵌入轻量级注意力模块 (如 1D-SE Block 通道注意力)。
   - 为了彻底探究“为什么高”，在 ResNet 中加入轻量级的 SE Attention 模块（或者直接提取最后一层卷积的类激活映射 CAM）。
   - 可视化：绘制“心电波形 + 注意力热力图 (Attention Heatmap)”。
   - 预期洞见：直观展示，深度学习不是在死记硬背数据，而是如同专业医生一样，将“注意力（红色的高权重区域）”精准汇聚在了心电图发生病变的特定波段上。
- **任务 4.2**：结合实验 3.5 的 Grad-CAM，针对各类异常心跳（室性早搏、房性早搏等），生成《基于深度学习模型的心电病理特征解释图册》。

### Phase 5: SOTA 模型引入与复现

不宽泛地使用通用 Transformer，而是复现使用时间序列分类领域被广泛引用的霸榜架构：**InceptionTime (Fawaz et al., 2020)**。

- **为什么选它？**  
  InceptionTime 是专门为 1D 时序设计的并行多尺度卷积架构。心电信号的物理特性是：既有时间极短的高频突变（QRS波，需极小卷积核），又有时间长且平缓的低频波（T波、P波，需大卷积核）。
- **复现任务**：实现包含并行大小不同 Kernel（如 10, 20, 40）的 1D Inception 模块，对比其与固定单一 Kernel 的 1D-ResNet 在捕获复杂 P-QRS-T 综合波时的特征差异。（可以直接调用 torch 库函数）

### Phase 6: 规范化的工程提交

- 模块化代码 (`dataset.py`, `models.py`, `trainer.py` 等)。
- 面向对象的架构设计与训练日志。

---

## 仓库结构
- `config/`               配置文件（训练/模型/数据）
- `data/`
  - `raw/`                原始数据
  - `processed/`          处理后数据
- `notebooks/`            Jupyter 笔记本
- `scripts/`              可执行脚本
- `src/`                  源代码包
  - `data/`               数据加载与处理模块
  - `evaluation/`         评估相关代码
  - `features/`           特征工程代码
  - `models/`             模型定义与接口
  - `training/`           训练脚本与训练管线
  - `utils/`              工具函数

- `README.md`, `HW2_TASKS.md`
