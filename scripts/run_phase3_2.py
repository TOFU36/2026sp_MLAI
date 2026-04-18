import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score as sk_f1_score, confusion_matrix

from src.utils.tools import set_seed, get_device, count_parameters, measure_latency
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D, BiLSTM, MLPMixer
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
    print("实验 3.2: 提取器归纳偏置对比 (5 trials + ANOVA)")
    print("=" * 50)

    # 从 Phase 1 复用 1D-ResNet 结果（训练配置一致：ResNet1D K=7, FocalLoss, lr=1e-3）
    phase1_json = 'results/logs/phase1_stats.json'
    if os.path.exists(phase1_json):
        with open(phase1_json) as f:
            phase1_data = json.load(f)
        resnet_scores = phase1_data['dl_test_f1']
        resnet_histories = phase1_data.get('dl_histories', [])
        print("\n>>> 1D-ResNet (reused from Phase 1)")
        print(f"  → Mean F1: {np.mean(resnet_scores):.4f}")
    else:
        raise FileNotFoundError(
            f"{phase1_json} not found — run Phase 1 first.")

    # 其他架构独立训练
    architectures = {
        "Bi-LSTM": lambda: BiLSTM(hidden_size=64),
        "MLP-Mixer": lambda: MLPMixer(seq_len=187),
    }

    all_scores = {"1D-ResNet": resnet_scores}
    all_histories = {"1D-ResNet": resnet_histories}
    summary = {
        "1D-ResNet": {
            'f1_mean': float(np.mean(resnet_scores)),
            'f1_std': float(np.std(resnet_scores)),
            'f1_scores': [float(s) for s in resnet_scores],
            'params': count_parameters(ResNet1D(kernel_size=7).to(device)),
            'latency_ms': float(measure_latency(ResNet1D(kernel_size=7).to(device), device)),
        },
    }

    for name, model_fn in architectures.items():
        scores, histories = [], []
        print(f"\n>>> {name}")
        for trial in range(N_TRIALS):
            print(f"  Trial {trial + 1}/{N_TRIALS}")
            set_seed(42 + trial)
            model = model_fn()
            train_loader, val_loader, test_loader = create_dataloaders(train_df, test_df)
            criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
            trainer = ECGTrainer(model, train_loader, val_loader, device,
                                 f'results/models/Phase3_2_{name}_T{trial}',
                                 criterion=criterion)
            hist = trainer.fit(epochs=100, lr=1e-3, weight_decay=1e-4)
            histories.append(hist)
            f1 = trainer.evaluate(test_loader)['f1']
            scores.append(f1)
            print(f"    Test F1: {f1:.4f}")

        all_scores[name] = scores
        all_histories[name] = histories
        mean_f1 = np.mean(scores)
        std_f1 = np.std(scores)

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
    posthoc = []
    if p_val < 0.05:
        print("→ 组间存在显著差异 (p < 0.05)")
        n_comp = len(names) * (len(names) - 1) // 2
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                t, p = stats.ttest_rel(groups[i], groups[j])
                p_adj = min(p * n_comp, 1.0)
                sig = p_adj < 0.05
                posthoc.append({'pair': f'{names[i]} vs {names[j]}',
                                't': float(t), 'p_adj': float(p_adj), 'significant': sig})
                marker = "*" if sig else ""
                print(f"  {names[i]} vs {names[j]}: p_adj={p_adj:.4e} {marker}")
    else:
        print("→ 组间无显著差异")

    # 最后一次训练的详细指标
    last_metrics = trainer.evaluate(test_loader)
    per_class_f1 = sk_f1_score(last_metrics['targets'], last_metrics['preds'], average=None).tolist()
    cm = confusion_matrix(last_metrics['targets'], last_metrics['preds']).tolist()

    save_json({
        **summary,
        'histories': all_histories,
        'anova_f': float(f_stat),
        'anova_p': float(p_val),
        'posthoc': posthoc,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'class_names': ['N', 'S', 'V', 'F', 'Q'],
    }, 'results/logs/phase3_2_bias.json')
    print("\nPhase 3.2 done.")


if __name__ == "__main__":
    main()
