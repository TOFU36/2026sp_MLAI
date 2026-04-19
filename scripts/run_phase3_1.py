import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import json
import torch
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score as sk_f1_score, confusion_matrix

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D, ResNet2D
from src.training.trainer import ECGTrainer
from src.training.losses import FocalLoss, compute_class_weights

N_TRIALS = 5


def train_or_load(model, trainer, test_loader, resume, **fit_kwargs):
    """根据 resume 参数决定跳过已有 checkpoint 还是重新训练。

    Args:
        resume: True 时若已有 best_model.pth 则加载跳过训练；False 时始终重新训练。
    Returns:
        (history, test_f1)
    """
    ckpt = os.path.join(trainer.save_dir, 'best_model.pth')
    if resume and os.path.exists(ckpt):
        print(f"  [skip] Loaded existing: {ckpt}")
        trainer.model.load_state_dict(torch.load(ckpt, weights_only=True))
        history = []
    else:
        history = trainer.fit(**fit_kwargs)

    metrics = trainer.evaluate(test_loader)
    return history, metrics['f1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='跳过已有 best_model.pth 的 trial，直接加载评估')
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test(sample_ratio=1.0)
    y_train_all = train_df.iloc[:, -1].values.astype(int)
    alpha_weights = compute_class_weights(y_train_all)

    print("=" * 50)
    print("实验 3.1: 输入模态对比 (5 trials + ANOVA)")
    if args.resume:
        print("  [resume mode] 跳过已有 checkpoint")
    print("=" * 50)

    # 从 Phase 1 复用 1D_Raw 结果
    phase1_json = 'results/logs/phase1_stats.json'
    if os.path.exists(phase1_json):
        with open(phase1_json) as f:
            phase1_data = json.load(f)
        raw_scores = phase1_data['dl_test_f1']
        raw_histories = phase1_data.get('dl_histories', [])
        print("\n>>> 1D_Raw (reused from Phase 1)")
        print(f"  → Mean F1: {np.mean(raw_scores):.4f}")
    else:
        raise FileNotFoundError(
            f"{phase1_json} not found — run Phase 1 first.")

    experiments = {
        "1D_FFT": 'fft',
        "2D_Mel": 'mel',
        "2D_CWT": 'cwt',
    }

    all_scores = {"1D_Raw": raw_scores}
    all_histories = {"1D_Raw": raw_histories}
    last_trainer = None
    last_test_loader = None

    for name, feat_type in experiments.items():
        scores, histories = [], []
        model_fn = ResNet1D if feat_type in ('raw', 'fft') else ResNet2D
        print(f"\n>>> {name} (feature: {feat_type})")
        for trial in range(N_TRIALS):
            print(f"  Trial {trial + 1}/{N_TRIALS}")
            set_seed(42 + trial)
            model = model_fn(in_channels=1, kernel_size=7) if feat_type != 'fft' else model_fn(in_channels=1)

            train_loader, val_loader, test_loader = create_dataloaders(
                train_df, test_df, feature_type=feat_type)
            criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
            trainer = ECGTrainer(model, train_loader, val_loader, device,
                                 f'results/models/Phase3_1_{name}_T{trial}',
                                 criterion=criterion)
            hist, f1 = train_or_load(
                model, trainer, test_loader, resume=args.resume,
                epochs=100, lr=1e-3, weight_decay=1e-4)
            histories.append(hist)
            scores.append(f1)
            print(f"    Test F1: {f1:.4f}")
            last_trainer = trainer
            last_test_loader = test_loader

        all_scores[name] = scores
        all_histories[name] = histories
        print(f"  → {name} Mean F1: {np.mean(scores):.4f}")

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

    last_metrics = last_trainer.evaluate(last_test_loader)
    per_class_f1 = sk_f1_score(last_metrics['targets'], last_metrics['preds'], average=None).tolist()
    cm = confusion_matrix(last_metrics['targets'], last_metrics['preds']).tolist()

    save_json({
        'scores': {k: [float(v) for v in vs] for k, vs in all_scores.items()},
        'means': {k: float(np.mean(vs)) for k, vs in all_scores.items()},
        'stds': {k: float(np.std(vs)) for k, vs in all_scores.items()},
        'histories': all_histories,
        'anova_f': float(f_stat),
        'anova_p': float(p_val),
        'posthoc': posthoc,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'class_names': ['N', 'S', 'V', 'F', 'Q'],
    }, 'results/logs/phase3_1_modality.json')
    print("\nPhase 3.1 done.")


if __name__ == "__main__":
    main()
