import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders, make_loader
from src.models import ResNet1D
from src.training.trainer import ECGTrainer

N_TRIALS = 5  # 多次试验用于统计分析，可调整为 3 快速验证


def run_anova_with_posthoc(group_scores, group_names):
    """单因素 ANOVA + Bonferroni 校正事后配对 T 检验"""
    groups = [group_scores[n] for n in group_names]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n  ANOVA: F={f_stat:.4f}, p={p_anova:.4e}")

    if p_anova < 0.05:
        print("  → 组间存在显著差异，进行 Bonferroni 事后检验:")
        n_comp = len(group_names) * (len(group_names) - 1) // 2
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                t, p = stats.ttest_rel(groups[i], groups[j])
                p_adj = min(p * n_comp, 1.0)
                sig = "显著" if p_adj < 0.05 else "不显著"
                mi, mj = np.mean(groups[i]), np.mean(groups[j])
                print(f"    {group_names[i]} vs {group_names[j]}: "
                      f"t={t:.3f}, p_adj={p_adj:.4e} ({sig})")
    else:
        print("  → 组间无显著差异")
    return f_stat, p_anova


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_df, test_df = load_train_test(sample_ratio=1.0)
    test_loader = make_loader(test_df)

    # ==================================================================
    # Task 2.1: 感受野 (Kernel Size) 动力学 — 多次试验 + ANOVA
    # ==================================================================
    print("\n" + "=" * 50)
    print("Task 2.1: Kernel Size Dynamics (5 trials + ANOVA)")
    print("=" * 50)

    kernel_sizes = [3, 15, 31]
    kernel_scores = {k: [] for k in kernel_sizes}
    kernel_histories = {k: [] for k in kernel_sizes}

    for k in kernel_sizes:
        print(f"\n>>> Kernel Size = {k}")
        for trial in range(N_TRIALS):
            print(f"  Trial {trial + 1}/{N_TRIALS}")
            set_seed(42 + trial)
            train_loader, val_loader = create_dataloaders(train_df, test_df)
            model = ResNet1D(in_channels=1, num_classes=5, kernel_size=k, use_se=False)
            trainer = ECGTrainer(model, train_loader, val_loader, device,
                                 f'results/models/Phase2_Kernel{k}_T{trial}')
            hist = trainer.fit(epochs=15, lr=1e-3, weight_decay=1e-4)

            f1 = trainer.evaluate(test_loader)['f1']
            kernel_scores[k].append(f1)
            kernel_histories[k].append(hist)
            print(f"    Test F1: {f1:.4f}")
        print(f"  → Kernel={k} Mean F1: {np.mean(kernel_scores[k]):.4f}")

    f_stat, p_anova = run_anova_with_posthoc(kernel_scores, [f"K={k}" for k in kernel_sizes])

    save_json({
        'kernel_sizes': kernel_sizes,
        'scores': {str(k): v for k, v in kernel_scores.items()},
        'histories': {str(k): v for k, v in kernel_histories.items()},
        'anova_f': float(f_stat),
        'anova_p': float(p_anova),
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

    for i, lr in enumerate(learning_rates):
        for j, wd in enumerate(weight_decays):
            print(f"\n>>> LR={lr}, WD={wd}")
            set_seed(42)
            train_loader, val_loader = create_dataloaders(train_df, test_df)
            model = ResNet1D(in_channels=1, num_classes=5, kernel_size=15, use_se=False)
            trainer = ECGTrainer(model, train_loader, val_loader, device,
                                 f'results/models/Phase2_Grid_LR{lr}_WD{wd}')
            trainer.fit(epochs=10, lr=lr, weight_decay=wd)
            f1_matrix[i][j] = float(trainer.evaluate(test_loader)['f1'])
            print(f"    Test F1: {f1_matrix[i][j]:.4f}")

    save_json({
        'learning_rates': learning_rates,
        'weight_decays': weight_decays,
        'f1_matrix': f1_matrix,
    }, 'results/logs/phase2_hyper_grid.json')
    print("\nPhase 2 done.")


if __name__ == "__main__":
    main()
