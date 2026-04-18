import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders, make_loader
from src.models import ResNet1D, ResNet2D
from src.training.trainer import ECGTrainer

N_TRIALS = 5


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test(sample_ratio=1.0)
    test_loader_raw = make_loader(test_df, feature_type='raw')
    test_loader_fft = make_loader(test_df, feature_type='fft')
    test_loader_mel = make_loader(test_df, feature_type='mel')
    test_loader_cwt = make_loader(test_df, feature_type='cwt')

    print("=" * 50)
    print("实验 3.1: 输入模态对比 (5 trials + ANOVA)")
    print("=" * 50)

    experiments = {
        "1D_Raw": ('raw', lambda: ResNet1D(in_channels=1, kernel_size=7), test_loader_raw),
        "1D_FFT": ('fft', lambda: ResNet1D(in_channels=1, kernel_size=7), test_loader_fft),
        "2D_Mel": ('mel', lambda: ResNet2D(in_channels=1), test_loader_mel),
        "2D_CWT": ('cwt', lambda: ResNet2D(in_channels=1), test_loader_cwt),
    }

    all_scores = {}
    for name, (feat_type, model_fn, t_loader) in experiments.items():
        scores = []
        print(f"\n>>> {name} (feature: {feat_type})")
        for trial in range(N_TRIALS):
            print(f"  Trial {trial + 1}/{N_TRIALS}")
            set_seed(42 + trial)
            model = model_fn()

            train_loader, val_loader = create_dataloaders(train_df, test_df, feature_type=feat_type)
            trainer = ECGTrainer(model, train_loader, val_loader, device,
                                 f'results/models/Phase3_1_{name}_T{trial}')
            trainer.fit(epochs=10)
            f1 = trainer.evaluate(t_loader)['f1']
            scores.append(f1)
            print(f"    Test F1: {f1:.4f}")

        all_scores[name] = scores
        print(f"  → {name} Mean F1: {np.mean(scores):.4f}")

    # ANOVA
    print("\n" + "-" * 40)
    names = list(all_scores.keys())
    groups = [all_scores[n] for n in names]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"ANOVA: F={f_stat:.4f}, p={p_val:.4e}")
    if p_val < 0.05:
        print("→ 组间存在显著差异 (p < 0.05)")
        n_comp = len(names) * (len(names) - 1) // 2
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                t, p = stats.ttest_rel(groups[i], groups[j])
                p_adj = min(p * n_comp, 1.0)
                sig = "*" if p_adj < 0.05 else ""
                print(f"  {names[i]} vs {names[j]}: p_adj={p_adj:.4e} {sig}")
    else:
        print("→ 组间无显著差异")

    save_json({k: [float(v) for v in vs] for k, vs in all_scores.items()},
              'results/logs/phase3_1_modality.json')
    print("\nPhase 3.1 done.")


if __name__ == "__main__":
    main()
