import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score as sk_f1_score, confusion_matrix

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D
from src.training.trainer import ECGTrainer
from src.training.losses import FocalLoss, compute_class_weights

N_TRIALS = 5


def run_anova_with_posthoc(group_scores, group_names):
    """单因素 ANOVA + Bonferroni 校正事后配对 T 检验"""
    groups = [group_scores[n] for n in group_names]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n  ANOVA: F={f_stat:.4f}, p={p_anova:.4e}")

    posthoc = []
    if p_anova < 0.05:
        print("  → 组间存在显著差异，进行 Bonferroni 事后检验:")
        n_comp = len(group_names) * (len(group_names) - 1) // 2
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                t, p = stats.ttest_rel(groups[i], groups[j])
                p_adj = min(p * n_comp, 1.0)
                sig = p_adj < 0.05
                posthoc.append({'pair': f'{group_names[i]} vs {group_names[j]}',
                                't': float(t), 'p_adj': float(p_adj), 'significant': sig})
                marker = "显著" if sig else "不显著"
                print(f"    {group_names[i]} vs {group_names[j]}: "
                      f"t={t:.3f}, p_adj={p_adj:.4e} ({marker})")
    else:
        print("  → 组间无显著差异")
    return f_stat, p_anova, posthoc


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_df, test_df = load_train_test(sample_ratio=1.0)
    y_train_all = train_df.iloc[:, -1].values.astype(int)
    alpha_weights = compute_class_weights(y_train_all)

    # ==================================================================
    # Task 2.1: 感受野 (Kernel Size) 动力学 — 多次试验 + ANOVA
    # ==================================================================
    print("\n" + "=" * 50)
    print("Task 2.1: Kernel Size Dynamics (5 trials + ANOVA)")
    print("=" * 50)

    # 从 Phase 1 复用 K=7 结果（训练配置一致：FocalLoss, lr=1e-3, wd=1e-4, epochs=100）
    phase1_json = 'results/logs/phase1_stats.json'
    if os.path.exists(phase1_json):
        with open(phase1_json) as f:
            phase1_data = json.load(f)
        k7_scores = phase1_data['dl_test_f1']
        print(f"\n>>> Kernel Size = 7 (reused from Phase 1)")
        print(f"  → K=7 Mean F1: {np.mean(k7_scores):.4f}")
    else:
        raise FileNotFoundError(
            f"{phase1_json} not found — run Phase 1 first.")

    kernel_sizes = [3, 15, 31]
    kernel_scores = {7: k7_scores}
    kernel_histories = {7: phase1_data.get('dl_histories', [])}

    for k in kernel_sizes:
        scores, histories = [], []
        print(f"\n>>> Kernel Size = {k}")
        for trial in range(N_TRIALS):
            print(f"  Trial {trial + 1}/{N_TRIALS}")
            set_seed(42 + trial)
            train_loader, val_loader, test_loader = create_dataloaders(train_df, test_df)
            model = ResNet1D(in_channels=1, num_classes=5, kernel_size=k, use_se=False)
            criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
            trainer = ECGTrainer(model, train_loader, val_loader, device,
                                 f'results/models/Phase2_Kernel{k}_T{trial}',
                                 criterion=criterion)
            hist = trainer.fit(epochs=100, lr=1e-3, weight_decay=1e-4)

            f1 = trainer.evaluate(test_loader)['f1']
            scores.append(f1)
            histories.append(hist)
            print(f"    Test F1: {f1:.4f}")
        kernel_scores[k] = scores
        kernel_histories[k] = histories
        print(f"  → Kernel={k} Mean F1: {np.mean(scores):.4f}")

    all_kernel_sizes = [7, 3, 15, 31]
    group_names = [f"K={k}" for k in all_kernel_sizes]
    f_stat, p_anova, posthoc = run_anova_with_posthoc(
        {f"K={k}": kernel_scores[k] for k in all_kernel_sizes}, group_names)

    # 最后一个 kernel 的最后一次 trial 的详细指标
    last_metrics = trainer.evaluate(test_loader)
    per_class_f1 = sk_f1_score(last_metrics['targets'], last_metrics['preds'], average=None).tolist()
    cm = confusion_matrix(last_metrics['targets'], last_metrics['preds']).tolist()

    save_json({
        'kernel_sizes': all_kernel_sizes,
        'scores': {str(k): [float(v) for v in vs] for k, vs in kernel_scores.items()},
        'means': {str(k): float(np.mean(vs)) for k, vs in kernel_scores.items()},
        'stds': {str(k): float(np.std(vs)) for k, vs in kernel_scores.items()},
        'histories': {str(k): v for k, v in kernel_histories.items()},
        'anova_f': float(f_stat),
        'anova_p': float(p_anova),
        'posthoc': posthoc,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'class_names': ['N', 'S', 'V', 'F', 'Q'],
    }, 'results/logs/phase2_kernel_dynamics.json')

    # ==================================================================
    # Task 2.2: 优化空间平滑性（网格搜索，单次运行）
    # ==================================================================
    print("\n" + "=" * 50)
    print("Task 2.2: LR vs Weight Decay Grid (单次运行)")
    print("=" * 50)

    learning_rates = [1e-2, 1e-3, 1e-4]
    weight_decays = [0.0, 1e-4, 1e-2]
    f1_matrix = np.zeros((len(learning_rates), len(weight_decays))).tolist()
    grid_histories = {}

    for i, lr in enumerate(learning_rates):
        for j, wd in enumerate(weight_decays):
            print(f"\n>>> LR={lr}, WD={wd}")
            set_seed(42)
            train_loader, val_loader, test_loader = create_dataloaders(train_df, test_df)
            model = ResNet1D(in_channels=1, num_classes=5, kernel_size=15, use_se=False)
            criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
            trainer = ECGTrainer(model, train_loader, val_loader, device,
                                 f'results/models/Phase2_Grid_LR{lr}_WD{wd}',
                                 criterion=criterion)
            hist = trainer.fit(epochs=20, lr=lr, weight_decay=wd)
            f1_matrix[i][j] = float(trainer.evaluate(test_loader)['f1'])
            grid_histories[f'LR{lr}_WD{wd}'] = hist
            print(f"    Test F1: {f1_matrix[i][j]:.4f}")

    save_json({
        'learning_rates': learning_rates,
        'weight_decays': weight_decays,
        'f1_matrix': f1_matrix,
        'histories': grid_histories,
    }, 'results/logs/phase2_hyper_grid.json')
    print("\nPhase 2 done.")


if __name__ == "__main__":
    main()
