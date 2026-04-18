import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.utils.tools import set_seed, get_device, count_parameters, measure_latency
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D, InceptionTime
from src.training.trainer import ECGTrainer
from src.training.losses import FocalLoss, compute_class_weights

N_TRIALS = 5


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test(sample_ratio=1.0)
    y_train_all = train_df.iloc[:, -1].values.astype(int)
    alpha_weights = compute_class_weights(y_train_all)

    print("=" * 50)
    print("Phase 5: InceptionTime vs ResNet1D (5 trials + T-test)")
    print("=" * 50)

    model_factories = {
        "ResNet1D": lambda: ResNet1D(kernel_size=7, use_se=True),
        "InceptionTime": lambda: InceptionTime(),
    }

    all_scores = {}
    all_histories = {}
    model_info = {}

    for name, model_fn in model_factories.items():
        scores = []
        histories = []
        print(f"\n>>> {name}")
        for trial in range(N_TRIALS):
            print(f"  Trial {trial + 1}/{N_TRIALS}")
            set_seed(42 + trial)
            model = model_fn()
            train_loader, val_loader, test_loader = create_dataloaders(train_df, test_df)
            criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
            trainer = ECGTrainer(model, train_loader, val_loader, device,
                                 f'results/models/Phase5_{name}_T{trial}',
                                 criterion=criterion)
            hist = trainer.fit(epochs=100, lr=1e-3, weight_decay=1e-4)
            f1 = trainer.evaluate(test_loader)['f1']
            scores.append(f1)
            histories.append(hist)
            print(f"    Test F1: {f1:.4f}")

        all_scores[name] = scores
        all_histories[name] = histories
        print(f"  → {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        # 参数量和延迟只需测一次
        sample_model = model_fn().to(device)
        model_info[name] = {
            'params': count_parameters(sample_model),
            'latency_ms': float(measure_latency(sample_model, device)),
        }

    # 配对 T 检验
    print("\n" + "-" * 40)
    names = list(all_scores.keys())
    t_stat, p_val = stats.ttest_rel(all_scores[names[0]], all_scores[names[1]])
    is_significant = p_val < 0.05
    print(f"Paired T-test: t={t_stat:.4f}, p={p_val:.4e}")
    if is_significant:
        winner = names[0] if np.mean(all_scores[names[0]]) > np.mean(all_scores[names[1]]) else names[1]
        print(f"→ {winner} 显著优于对方 (p < 0.05)")
    else:
        print("→ 两者无显著差异")

    save_json({
        'scores': {k: [float(s) for s in v] for k, v in all_scores.items()},
        'means': {k: float(np.mean(v)) for k, v in all_scores.items()},
        'stds': {k: float(np.std(v)) for k, v in all_scores.items()},
        'histories': all_histories,
        'model_info': model_info,
        't_stat': float(t_stat),
        'p_val': float(p_val),
        'is_significant': is_significant,
    }, 'results/logs/phase5_sota_comparison.json')
    print("\nPhase 5 done.")


if __name__ == "__main__":
    main()
