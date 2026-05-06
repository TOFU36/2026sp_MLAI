import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from scipy import stats

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D
from src.training.trainer import ECGTrainer
from src.training.losses import FocalLoss, compute_class_weights

N_TRIALS = 5


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_df, test_df = load_train_test(sample_ratio=1.0)
    y_train_all = train_df.iloc[:, -1].values.astype(int)
    X_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values.astype(int)
    alpha_weights = compute_class_weights(y_train_all)

    ml_test_scores, dl_test_scores = [], []
    dl_histories = []

    print("\n" + "=" * 50)
    print(f"Phase 1: ML vs DL ({N_TRIALS} trials + 配对 T 检验)")
    print("=" * 50)

    for trial in range(N_TRIALS):
        print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")

        # ---- ML: RF + SMOTE ----
        print("[ML]: RF + SMOTE ...")
        set_seed(42 + trial)
        X_res, y_res = SMOTE(random_state=42 + trial).fit_resample(
            train_df.iloc[:, :-1].values, y_train_all)
        rf = RandomForestClassifier(n_estimators=100, random_state=42 + trial, n_jobs=-1)
        rf.fit(X_res, y_res)
        ml_f1 = f1_score(y_test, rf.predict(X_test), average='macro')
        ml_test_scores.append(ml_f1)
        print(f"  -> ML Test F1: {ml_f1:.4f}")

        # ---- DL: 1D-ResNet + Focal Loss ----
        print("[DL]: 1D-ResNet + Focal Loss ...")
        set_seed(42 + trial)
        train_loader, val_loader, test_loader = create_dataloaders(train_df, test_df)
        model = ResNet1D(in_channels=1, num_classes=5, kernel_size=7, use_se=False)
        criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
        trainer = ECGTrainer(model, train_loader, val_loader, device,
                             f'results/models/Phase1_ResNet_T{trial}', criterion=criterion)
        hist = trainer.fit(epochs=100, lr=1e-3, weight_decay=1e-4)
        dl_histories.append(hist)

        dl_f1 = trainer.evaluate(test_loader)['f1']
        dl_test_scores.append(dl_f1)
        print(f"  -> DL Test F1: {dl_f1:.4f}")

    # 统计检验
    print("\n" + "=" * 50)
    print("配对 T 检验（测试集 F1）")
    print("=" * 50)
    print(f"ML: {np.mean(ml_test_scores):.4f} ± {np.std(ml_test_scores):.4f}")
    print(f"DL: {np.mean(dl_test_scores):.4f} ± {np.std(dl_test_scores):.4f}")

    t_stat, p_val = stats.ttest_rel(dl_test_scores, ml_test_scores)
    print(f"Paired T-test: t={t_stat:.4f}, p={p_val:.4e}")

    if p_val < 0.05:
        winner = "DL" if np.mean(dl_test_scores) > np.mean(ml_test_scores) else "ML"
        print(f"结论: {winner} 显著优于对方 (p < 0.05)")
    else:
        print("结论: 两者无显著差异。")

    # 最后一个 trial 的详细指标（混淆矩阵、每类 F1）
    last_metrics = trainer.evaluate(test_loader)
    per_class_f1 = f1_score(last_metrics['targets'], last_metrics['preds'], average=None).tolist()
    cm = confusion_matrix(last_metrics['targets'], last_metrics['preds']).tolist()

    save_json({
        'ml_test_f1': [float(v) for v in ml_test_scores],
        'dl_test_f1': [float(v) for v in dl_test_scores],
        'ml_mean': float(np.mean(ml_test_scores)),
        'ml_std': float(np.std(ml_test_scores)),
        'dl_mean': float(np.mean(dl_test_scores)),
        'dl_std': float(np.std(dl_test_scores)),
        't_stat': float(t_stat),
        'p_val': float(p_val),
        'is_significant': bool(p_val < 0.05),
        'dl_histories': dl_histories,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'class_names': ['N', 'S', 'V', 'F', 'Q'],
    }, 'results/logs/phase1_stats.json')


if __name__ == "__main__":
    main()
