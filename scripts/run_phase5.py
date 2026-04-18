import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders, make_loader
from src.models import ResNet1D, InceptionTime
from src.training.trainer import ECGTrainer

N_TRIALS = 5


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test(sample_ratio=1.0)
    train_loader, test_loader = create_dataloaders(train_df, test_df)

    print("=" * 50)
    print("Phase 5: InceptionTime vs ResNet1D (5 trials + T-test)")
    print("=" * 50)

    model_factories = {
        "ResNet1D": lambda: ResNet1D(kernel_size=7, use_se=True),
        "InceptionTime": lambda: InceptionTime(),
    }

    all_scores = {}
    all_histories = {}

    for name, model_fn in model_factories.items():
        scores = []
        histories = []
        print(f"\n>>> {name}")
        for trial in range(N_TRIALS):
            print(f"  Trial {trial + 1}/{N_TRIALS}")
            set_seed(42 + trial)
            model = model_fn()
            trainer = ECGTrainer(model, train_loader, test_loader, device,
                                 f'results/models/Phase5_{name}_T{trial}')
            hist = trainer.fit(epochs=20, lr=1e-3, weight_decay=1e-4)
            f1 = trainer.evaluate(test_loader)['f1']
            scores.append(f1)
            histories.append(hist)
            print(f"    Test F1: {f1:.4f}")

        all_scores[name] = scores
        all_histories[name] = histories
        print(f"  → {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # 配对 T 检验
    print("\n" + "-" * 40)
    names = list(all_scores.keys())
    t_stat, p_val = stats.ttest_rel(all_scores[names[0]], all_scores[names[1]])
    print(f"Paired T-test: t={t_stat:.4f}, p={p_val:.4e}")
    if p_val < 0.05:
        winner = names[0] if np.mean(all_scores[names[0]]) > np.mean(all_scores[names[1]]) else names[1]
        print(f"→ {winner} 显著优于对方 (p < 0.05)")
    else:
        print("→ 两者无显著差异")

    save_json({
        k: {'scores': [float(s) for s in v],
            'mean': float(np.mean(v)),
            'std': float(np.std(v))}
        for k, v in all_scores.items()
    }, 'results/logs/phase5_sota_comparison.json')
    print("\nPhase 5 done.")


if __name__ == "__main__":
    main()
