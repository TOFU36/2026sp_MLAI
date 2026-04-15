import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from src.utils.tools import set_seed
from src.utils.logger import save_json
from src.data.dataset import ECGDataset
from src.models.networks import ResNet1D
from src.training.trainer import ECGTrainer

def get_embeddings(model, loader, device):
    """抽取特征层的 Embedding"""
    model.eval()
    embs, labels = [],[]
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            # 使用前面在 networks.py 中预留的 extract_features 接口
            features = model.extract_features(X_batch)
            embs.append(features.cpu().numpy())
            labels.append(y_batch.numpy())
    return np.concatenate(embs), np.concatenate(labels)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/raw/mitbih_train_downsampled_3000.csv', header=None).fillna(0)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:, -1], random_state=42)

    print("="*50 + "\n⚔️ 实验 3.3: 分类决策层解耦 (Softmax vs ML)\n" + "="*50)

    train_loader = DataLoader(ECGDataset(train_df, feature_type='raw'), batch_size=64, shuffle=True)
    test_loader = DataLoader(ECGDataset(test_df, feature_type='raw'), batch_size=64, shuffle=False)

    # 1. 训练一个基线特征提取器
    print(">>> Pre-training the 1D-ResNet Feature Extractor...")
    model = ResNet1D(use_se=False)
    trainer = ECGTrainer(model, train_loader, test_loader, device, 'results/models/Phase3_3_Extractor')
    trainer.fit(epochs=10) # 快速训练以获得特征空间

    # 获取基线 Softmax 的最终分数
    base_metrics = trainer.evaluate()
    
    # 2. 提取 Embeddings
    print(">>> Extracting Embeddings for ML classifiers...")
    X_train_emb, y_train_emb = get_embeddings(model, train_loader, device)
    X_test_emb, y_test_emb = get_embeddings(model, test_loader, device)

    # 3. 对比传统 ML 决策层
    classifiers = {
        "End-to-End Softmax": base_metrics['f1'],
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', class_weight='balanced', random_state=42)
    }

    results = {}
    for name, clf in classifiers.items():
        if name == "End-to-End Softmax":
            results[name] = clf
            print(f"-> {name} F1-Macro: {clf:.4f}")
        else:
            print(f">>> Fitting {name} on 1D-CNN Embeddings...")
            clf.fit(X_train_emb, y_train_emb)
            preds = clf.predict(X_test_emb)
            score = f1_score(y_test_emb, preds, average='macro')
            results[name] = score
            print(f"-> {name} F1-Macro: {score:.4f}")

    save_json(results, 'results/logs/phase3_3_classifiers.json')
    print("\nPhase 3.3 实验完成，结果已保存！")

if __name__ == "__main__":
    main()