import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from scipy import stats

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, make_loader
from src.models import ResNet1D
from src.evaluation.embeddings import get_embeddings

N_TRIALS = 5


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test(sample_ratio=1.0)

    # 训练集全量用于提取 embedding（不分割，因为不训练）
    train_loader = make_loader(train_df, shuffle=False)
    test_loader = make_loader(test_df)

    print("=" * 50)
    print("实验 3.3: 分类器解耦对比 (5 trials + ANOVA)")
    print("=" * 50)

    softmax_scores, rf_scores, svm_scores = [], [], []

    for trial in range(N_TRIALS):
        print(f"\n>>> Trial {trial + 1}/{N_TRIALS}")
        set_seed(42 + trial)

        # 加载 Phase 1 已训练好的模型（ResNet1D K=7, FocalLoss, lr=1e-3）
        ckpt_path = f'results/models/Phase1_ResNet_T{trial}/best_model.pth'
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"{ckpt_path} not found — run Phase 1 first.")

        model = ResNet1D(kernel_size=7, use_se=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print(f"  Loaded {ckpt_path}")

        # Softmax 测试集 F1（直接用已训练好的分类头）
        from src.training.trainer import ECGTrainer
        trainer = ECGTrainer(model, train_loader, test_loader, device,
                             f'results/models/Phase3_3_T{trial}')
        softmax_f1 = trainer.evaluate(test_loader)['f1']
        softmax_scores.append(softmax_f1)

        # 提取 Embeddings
        X_tr_emb, y_tr_emb = get_embeddings(model, train_loader, device)
        X_te_emb, y_te_emb = get_embeddings(model, test_loader, device)

        # RF
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_tr_emb, y_tr_emb)
        rf_f1 = f1_score(y_te_emb, rf.predict(X_te_emb), average='macro')
        rf_scores.append(rf_f1)

        # SVM
        svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        svm.fit(X_tr_emb, y_tr_emb)
        svm_f1 = f1_score(y_te_emb, svm.predict(X_te_emb), average='macro')
        svm_scores.append(svm_f1)

        print(f"  Softmax: {softmax_f1:.4f}  RF: {rf_f1:.4f}  SVM: {svm_f1:.4f}")

    # 汇总
    all_scores = {"Softmax": softmax_scores, "RF": rf_scores, "SVM": svm_scores}
    for name, scores in all_scores.items():
        print(f"  {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # ANOVA
    print("\n" + "-" * 40)
    names = list(all_scores.keys())
    groups = list(all_scores.values())
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"ANOVA: F={f_stat:.4f}, p={p_val:.4e}")
    posthoc = []
    if p_val < 0.05:
        print("→ 组间存在显著差异 (p < 0.05)")
        n_comp = 3
        for i in range(3):
            for j in range(i + 1, 3):
                t, p = stats.ttest_rel(groups[i], groups[j])
                p_adj = min(p * n_comp, 1.0)
                sig = p_adj < 0.05
                posthoc.append({'pair': f'{names[i]} vs {names[j]}',
                                't': float(t), 'p_adj': float(p_adj), 'significant': sig})
                marker = "*" if sig else ""
                print(f"  {names[i]} vs {names[j]}: p_adj={p_adj:.4e} {marker}")
    else:
        print("→ 组间无显著差异")

    # 最后一次试验的详细指标
    last_metrics = trainer.evaluate(test_loader)
    cm = confusion_matrix(last_metrics['targets'], last_metrics['preds']).tolist()

    save_json({
        'scores': {k: [float(v) for v in vs] for k, vs in all_scores.items()},
        'means': {k: float(np.mean(vs)) for k, vs in all_scores.items()},
        'stds': {k: float(np.std(vs)) for k, vs in all_scores.items()},
        'anova_f': float(f_stat),
        'anova_p': float(p_val),
        'posthoc': posthoc,
        'confusion_matrix': cm,
        'class_names': ['N', 'S', 'V', 'F', 'Q'],
    }, 'results/logs/phase3_3_classifiers.json')
    print("\nPhase 3.3 done.")


if __name__ == "__main__":
    main()
