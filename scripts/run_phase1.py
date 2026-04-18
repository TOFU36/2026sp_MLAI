import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from scipy import stats

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders, make_loader
from src.models import ResNet1D
from src.training.trainer import ECGTrainer
from src.training.losses import FocalLoss


def compute_class_weights(y):
    """计算逆频率类别权重（sklearn balanced 风格）"""
    counts = np.bincount(y)
    n_classes = len(counts)
    return [len(y) / (n_classes * c) for c in counts]


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 加载数据：训练集 + 固定测试集（所有折共享同一测试集，确保评估无偏）
    train_df, test_df = load_train_test(sample_ratio=1.0)
    y_train_all = train_df.iloc[:, -1].values.astype(int)
    alpha_weights = compute_class_weights(y_train_all)

    X_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values.astype(int)
    test_loader = make_loader(test_df)

    # ================================================================
    # 5折交叉验证：每折训练不同的子集，统一在固定测试集上评估
    # - CV 验证折仅用于 DL 的 early stopping / 模型选择
    # ================================================================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ml_test_scores, dl_test_scores = [], []

    print("\n" + "=" * 50)
    print("Phase 1: 5-Fold CV → 统一测试集评估 → 配对 T 检验")
    print("=" * 50)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y_train_all)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)  # 仅用于 DL early stopping

        # ---- ML: RF + SMOTE ----
        print("[ML]: RF + SMOTE ...")
        X_tr, y_tr = tr_df.iloc[:, :-1].values, tr_df.iloc[:, -1].values.astype(int)
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_res, y_res)
        ml_f1 = f1_score(y_test, rf.predict(X_test), average='macro')
        ml_test_scores.append(ml_f1)
        print(f"  -> ML Test F1: {ml_f1:.4f}")

        # ---- DL: 1D-ResNet + Focal Loss ----
        print("[DL]: 1D-ResNet + Focal Loss ...")
        set_seed(42 + fold)
        train_loader, val_loader = create_dataloaders(tr_df, val_df)
        model = ResNet1D(in_channels=1, num_classes=5, kernel_size=7, use_se=False)
        criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
        trainer = ECGTrainer(model, train_loader, val_loader, device,
                             f'results/models/Phase1_ResNet_Fold{fold}', criterion=criterion)
        trainer.fit(epochs=100, lr=1e-3, weight_decay=1e-4)

        # fit() 结束后 trainer 已自动加载最优权重，直接在测试集上评估
        dl_f1 = trainer.evaluate(test_loader)['f1']
        dl_test_scores.append(dl_f1)
        print(f"  -> DL Test F1: {dl_f1:.4f}")

    # ================================================================
    # 配对 T 检验（基于测试集 F1，无偏）
    # ================================================================
    print("\n" + "=" * 50)
    print("配对 T 检验（测试集 F1）")
    print("=" * 50)
    print(f"ML Test F1: {[round(x, 4) for x in ml_test_scores]} | Mean: {np.mean(ml_test_scores):.4f}")
    print(f"DL Test F1: {[round(x, 4) for x in dl_test_scores]} | Mean: {np.mean(dl_test_scores):.4f}")

    t_stat, p_val = stats.ttest_rel(dl_test_scores, ml_test_scores)
    print(f"Paired T-test: t={t_stat:.4f}, p={p_val:.4e}")

    if p_val < 0.05:
        winner = "DL (1D-ResNet)" if np.mean(dl_test_scores) > np.mean(ml_test_scores) else "ML (RF+SMOTE)"
        print(f"结论: {winner} 在统计学上显著优于对方 (p < 0.05)")
    else:
        print("结论: 两者无显著差异。")

    save_json({
        'ml_test_f1': ml_test_scores,
        'dl_test_f1': dl_test_scores,
        't_stat': float(t_stat),
        'p_val': float(p_val),
        'is_significant': bool(p_val < 0.05),
    }, 'results/logs/phase1_stats.json')


if __name__ == "__main__":
    main()
