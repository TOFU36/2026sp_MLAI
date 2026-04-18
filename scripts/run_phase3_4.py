import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders, make_loader
from src.data.dataset import ECGDataset
from src.models import ResNet1D
from src.training.trainer import ECGTrainer
from src.training.losses import FocalLoss, compute_class_weights


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()

    y_train_all = train_df.iloc[:, -1].values.astype(int)
    alpha_weights = compute_class_weights(y_train_all)

    clean_test_loader = make_loader(test_df)

    # Build noisy test set
    noisy_test_ds = ECGDataset(test_df)
    noisy_test_ds.X = noisy_test_ds.X + 0.1 * np.random.randn(*noisy_test_ds.X.shape)
    from torch.utils.data import DataLoader
    noisy_test_loader = DataLoader(noisy_test_ds, batch_size=256, shuffle=False,
                                   num_workers=4, pin_memory=True)

    print("=" * 50 + "\nExperiment 3.4: Domain-Specific Augmentation\n" + "=" * 50)

    # Baseline (no augmentation): 复用 Phase 1 的 trial 0 结果
    phase1_json = 'results/logs/phase1_stats.json'
    if os.path.exists(phase1_json):
        with open(phase1_json) as f:
            phase1_data = json.load(f)
        base_clean_f1 = phase1_data['dl_test_f1'][0]
        base_history = phase1_data['dl_histories'][0]
        print(f"\n>>> Baseline (reused from Phase 1 trial 0)")
        print(f"  Clean F1: {base_clean_f1:.4f}")
    else:
        raise FileNotFoundError(
            f"{phase1_json} not found — run Phase 1 first.")

    # Baseline 在 noisy test 上的评估：加载 Phase 1 trial 0 的模型
    ckpt_path = 'results/models/Phase1_ResNet_T0/best_model.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found — run Phase 1 first.")
    base_model = ResNet1D(kernel_size=7, use_se=False).to(device)
    base_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    trainer_base = ECGTrainer(base_model, clean_test_loader, clean_test_loader, device,
                              'results/models/Phase3_4_Base')
    base_noisy_f1 = trainer_base.evaluate(noisy_test_loader)['f1']
    print(f"  Noisy F1: {base_noisy_f1:.4f}")

    # Train augmented model（训练集带增强，验证集/测试集不带）
    print("\n>>> Augmented model ...")
    train_loader_aug, val_loader, _ = create_dataloaders(train_df, test_df, augment=True)
    aug_model = ResNet1D(kernel_size=7, use_se=False)
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    trainer_aug = ECGTrainer(aug_model, train_loader_aug, val_loader, device,
                             'results/models/Phase3_4_Aug', criterion=criterion)
    aug_history = trainer_aug.fit(epochs=100, lr=1e-3, weight_decay=1e-4)

    # Evaluate augmented model on test set
    aug_clean = trainer_aug.evaluate(clean_test_loader)['f1']
    aug_noisy = trainer_aug.evaluate(noisy_test_loader)['f1']

    results = {
        "Baseline_Clean_F1": base_clean_f1,
        "Baseline_Noisy_F1": base_noisy_f1,
        "Baseline_Drop": base_clean_f1 - base_noisy_f1,
        "Augmented_Clean_F1": aug_clean,
        "Augmented_Noisy_F1": aug_noisy,
        "Augmented_Drop": aug_clean - aug_noisy,
        "baseline_history": base_history,
        "aug_history": aug_history,
    }

    print(f"\n[Robustness] Baseline drop: {results['Baseline_Drop']:.4f}, "
          f"Augmented drop: {results['Augmented_Drop']:.4f}")

    save_json(results, 'results/logs/phase3_4_augmentation.json')
    print("\nPhase 3.4 done.")


if __name__ == "__main__":
    main()
