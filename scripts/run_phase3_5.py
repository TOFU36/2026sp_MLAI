import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.utils.tools import set_seed
from src.utils.logger import save_json
from src.data.dataset import ECGDataset
from src.models.networks import ResNet1D
from src.training.trainer import ECGTrainer

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/raw/mitbih_train_downsampled_3000.csv', header=None).fillna(0)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:, -1], random_state=42)

    print("="*50 + "\n⌚ 实验 3.5: 可穿戴设备 R 波对齐漂移抗性\n" + "="*50)

    # 1. 准备基线模型
    print(">>> Preparing 1D-CNN and Random Forest models...")
    # CNN
    train_loader = DataLoader(ECGDataset(train_df, feature_type='raw'), batch_size=64, shuffle=True)
    cnn_model = ResNet1D()
    trainer = ECGTrainer(cnn_model, train_loader, None, device, 'results/models/Phase3_5_CNN')
    trainer.fit(epochs=8) # 快速训练
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values.astype(int))

    # 2. 注入随机抖动并测试
    jitters =[0, 2, 5, 10, 15]
    cnn_scores, ml_scores = [],[]
    
    for j in jitters:
        print(f"\n>>> Testing Jitter Offset = +/- {j} steps")
        # 为测试集引入均匀随机抖动 (Jitter)
        jitter_test_dataset = ECGDataset(test_df, feature_type='raw', max_jitter=j)
        jitter_test_loader = DataLoader(jitter_test_dataset, batch_size=64, shuffle=False)
        
        # 测试 CNN
        metrics = trainer.evaluate(jitter_test_loader)
        cnn_scores.append(metrics['f1'])
        
        # 测试 RF
        X_jitter_test = jitter_test_dataset.X # 数据集内部会施加 jitter
        y_test = jitter_test_dataset.y
        rf_preds = rf_model.predict(X_jitter_test)
        
        from sklearn.metrics import f1_score
        rf_f1 = f1_score(y_test, rf_preds, average='macro')
        ml_scores.append(rf_f1)
        
        print(f"    1D-CNN F1: {metrics['f1']:.4f}  |  RF F1: {rf_f1:.4f}")

    results = {
        'jitters': jitters,
        'cnn_scores': cnn_scores,
        'ml_scores': ml_scores
    }
    
    save_json(results, 'results/logs/phase3_5_jitter_robustness.json')
    print("\nPhase 3.5 实验完成，结果已保存！请至 Notebook 读取数据并绘制平移崩溃曲线。")

if __name__ == "__main__":
    main()