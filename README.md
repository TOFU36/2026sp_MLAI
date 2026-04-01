# 项目目录结构

本项目已重建为以下结构：

- config/               配置文件目录（Python 包）
- data/
  - raw/                原始数据
  - processed/          处理后数据
- notebooks/            Jupyter 笔记本
- scripts/              可执行脚本
- src/                  源代码包
  - data/               数据加载与处理模块
  - evaluation/         评估相关代码
  - features/           特征工程代码
  - models/             模型定义与接口
  - training/           训练脚本与训练管线
  - utils/              工具函数

说明：
- 各 Python 包目录包含空的 `__init__.py` 以便于作为包导入。
- 将来可将数据放入 `data/raw`，并将清洗/预处理结果放入 `data/processed`。