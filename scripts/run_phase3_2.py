import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.utils.tools import set_seed, get_device, count_parameters, measure_latency
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders, make_loader
from src.models import ResNet1D, BiLSTM, MLPMixer
from src.training.trainer import ECGTrainer

N_TRIALS = 5


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test(sample_ratio=1.0)
    test_loader = make_loader(test_df)

    print("=" * 50)
    print("实验 3.2: 提取器归纳偏置对比 (5 trials + ANOVA)")
    print("=" * 50)

    architectures = {
        "1D-ResNet": lambda: ResNet1D(kernel_size=7),
        "Bi-LSTM": lambda: BiLSTM(hidden_size=64),
        "MLP-Mixer": lambda: MLPMixer(seq_len=187),
    }

    all_scores = {}
    summary = {}

    for name, model_fn in architectures.items():
        scores = []
        print(f"\n>>> {name}")
        for trial in range(N_TRIALS):
            print(f"  Trial {trial + 1}/{N_TRIALS}")
            set_seed(42 + trial)
            model = model_fn()
            train_loader, val_loader = create_dataloaders(train_df, test_df)
            trainer = ECGTrainer(model, train_loader, val_loader, device,
                                 f'results/models/Phase3_2_{name}_T{trial}')
            trainer.fit(epochs=10)
            f1 = trainer.evaluate(test_loader)['f1']
            scores.append(f1)
            print(f"    Test F1: {f1:.4f}")

        all_scores[name] = scores
        mean_f1 = np.mean(scores)
        std_f1 = np.std(scores)

        # 参数量和延迟只需测一次
        sample_model = model_fn().to(device)
        params = count_parameters(sample_model)
        latency = measure_latency(sample_model, device)

        summary[name] = {
            'f1_mean': float(mean_f1),
            'f1_std': float(std_f1),
            'f1_scores': [float(s) for s in scores],
            'params': params,
            'latency_ms': float(latency),
        }
        print(f"  → {name}: F1={mean_f1:.4f}±{std_f1:.4f}, Params={params}, Latency={latency:.2f}ms")

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

    save_json(summary, 'results/logs/phase3_2_bias.json')
    print("\nPhase 3.2 done.")


if __name__ == "__main__":
    main()
