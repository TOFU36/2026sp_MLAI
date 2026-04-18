import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from scipy import stats

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders, make_loader
from src.models import ResNet1D
from src.training.trainer import ECGTrainer
from src.evaluation.embeddings import get_embeddings

N_TRIALS = 5


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test(sample_ratio=1.0)
    train_loader, test_loader = create_dataloaders(train_df, test_df)

    print("=" * 50)
    print("实验 3.3: 分类器解耦对比 (5 trials + ANOVA)")
    print("=" * 50)

    softmax_scores = []
    rf_scores = []
    svm_scores = []

    for trial in range(N_TRIALS):
        print(f"\n>>> Trial {trial + 1}/{N_TRIALS}")
        set_seed(42 + trial)

        # 训练特征提取器
        model = ResNet1D(use_se=False)
        trainer = ECGTrainer(model, train_loader, test_loader, device,
                             f'results/models/Phase3_3_T{trial}')
        trainer.fit(epochs=10)

        # Softmax 测试集 F1
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
    if p_val < 0.05:
        print("→ 组间存在显著差异 (p < 0.05)")
        n_comp = 3
        for i in range(3):
            for j in range(i + 1, 3):
                t, p = stats.ttest_rel(groups[i], groups[j])
                p_adj = min(p * n_comp, 1.0)
                sig = "*" if p_adj < 0.05 else ""
                print(f"  {names[i]} vs {names[j]}: p_adj={p_adj:.4e} {sig}")
    else:
        print("→ 组间无显著差异")

    save_json({k: [float(v) for v in vs] for k, vs in all_scores.items()},
              'results/logs/phase3_3_classifiers.json')
    print("\nPhase 3.3 done.")


if __name__ == "__main__":
    main()
