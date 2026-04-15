import sys
import os
# 将项目根目录添加到系统路径，以支持 src 的导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from scipy import stats

from src.utils.tools import set_seed
from src.utils.logger import save_json
from src.data.dataset import ECGDataset
from src.models.networks import ResNet1D
from src.training.trainer import ECGTrainer
from src.training.losses import FocalLoss

def main():
    # 强制固定全局随机种子，确保 DL 与 ML 实验的可重复性
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    data_path = 'data/raw/mitbih_train_downsampled_3000.csv'
    try:
        df = pd.read_csv(data_path, header=None)
    except FileNotFoundError:
        print("未找到数据，生成虚拟数据用于代码测试流程...")
        X_dummy = np.random.randn(3000, 187) * 0.1
        y_dummy = np.random.choice([0, 1, 2, 3, 4], size=3000, p=[0.8, 0.05, 0.08, 0.02, 0.05])
        df = pd.DataFrame(X_dummy)
        df[187] = y_dummy

    # 预处理确保数据类型正确
    X_df = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').fillna(0)
    df.iloc[:, :-1] = X_df
    y_full = df.iloc[:, -1].values.astype(int)

    # 计算类别频率的反比作为 Loss 的 Alpha 权重
    class_counts = np.bincount(y_full)
    total_samples = len(y_full)
    alpha_weights = [total_samples / c for c in class_counts]
    # 归一化权重防止 loss 过大
    alpha_weights =[w / sum(alpha_weights) for w in alpha_weights] 
    
    # 2. 5折交叉验证设置 (Stratified K-Fold 保持每折不平衡比例一致)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    ml_f1_scores =[]
    dl_f1_scores =[]
    
    print("\n" + "="*50)
    print("🚀 Phase 1: 5-Fold Cross Validation - ML vs DL")
    print("="*50)

    for fold, (train_idx, test_idx) in enumerate(skf.split(df, y_full)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        # ---------------------------------------------------------
        # [Pipeline A]: 传统 ML (RF + SMOTE) 作为强基线
        # ---------------------------------------------------------
        print("[ML Pipeline]: Training Random Forest with SMOTE...")
        X_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values.astype(int)
        X_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values.astype(int)
        
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_res, y_train_res)
        ml_preds = rf.predict(X_test)
        ml_f1 = f1_score(y_test, ml_preds, average='macro')
        ml_f1_scores.append(ml_f1)
        print(f"-> ML (RF+SMOTE) F1-Macro: {ml_f1:.4f}")

        # ---------------------------------------------------------
        # [Pipeline B]: 深度学习 DL (1D-ResNet + Focal Loss)
        # ---------------------------------------------------------
        print("[DL Pipeline]: Training 1D-ResNet with Focal Loss...")
        # 重置种子以确保每一折 DL 网络初始化的公平性
        set_seed(42 + fold) 
        
        train_loader = DataLoader(ECGDataset(train_df, feature_type='raw'), batch_size=64, shuffle=True)
        test_loader = DataLoader(ECGDataset(test_df, feature_type='raw'), batch_size=64, shuffle=False)
        
        model = ResNet1D(in_channels=1, num_classes=5, kernel_size=7, use_se=False)
        criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
        save_dir = f'results/models/Phase1_ResNet_Fold{fold}'
        
        trainer = ECGTrainer(model, train_loader, test_loader, device, save_dir, criterion=criterion)
        # 训练 15 个 epoch
        hist = trainer.fit(epochs=15, lr=1e-3) 
        
        best_dl_f1 = max(hist['val_f1'])
        dl_f1_scores.append(best_dl_f1)
        print(f"-> DL (1D-ResNet) Best F1-Macro: {best_dl_f1:.4f}")

    # 3. 统计学分析 (配对 T 检验)
    print("\n" + "="*50)
    print("📊 Statistical Comparison Results")
    print("="*50)
    print(f"ML F1-Macro Scores: {[round(x,4) for x in ml_f1_scores]} | Mean: {np.mean(ml_f1_scores):.4f}")
    print(f"DL F1-Macro Scores: {[round(x,4) for x in dl_f1_scores]} | Mean: {np.mean(dl_f1_scores):.4f}")
    
    t_stat, p_val = stats.ttest_rel(dl_f1_scores, ml_f1_scores)
    print(f"Paired T-test: t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")
    
    is_significant = p_val < 0.05
    if is_significant:
        if np.mean(dl_f1_scores) > np.mean(ml_f1_scores):
            print("结论: 深度学习(1D-ResNet) 在统计学上 **显著优于** 传统机器学习(RF+SMOTE)！")
        else:
            print("结论: 传统机器学习(RF+SMOTE) 在统计学上 **显著优于** 深度学习(1D-ResNet)！")
    else:
        print("结论: 两者在统计学上 **没有显著差异**。")

    # 4. 保存实验结果供 Notebook 绘图调用
    results_to_save = {
        'ml_f1_scores': ml_f1_scores,
        'dl_f1_scores': dl_f1_scores,
        't_stat': float(t_stat),
        'p_val': float(p_val),
        'is_significant': is_significant
    }
    save_json(results_to_save, 'results/logs/phase1_stats.json')
    print("-> Phase 1 结果已持久化保存至 results/logs/phase1_stats.json")

if __name__ == "__main__":
    main()