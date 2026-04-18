import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders, make_loader
from src.data.dataset import ECGDataset
from src.models import ResNet1D
from src.training.trainer import ECGTrainer


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()

    clean_train_loader = make_loader(train_df, shuffle=True)
    aug_train_loader = make_loader(train_df, shuffle=True, augment=True)
    clean_test_loader = make_loader(test_df)

    # Build noisy test set
    noisy_test_ds = ECGDataset(test_df)
    noisy_test_ds.X = noisy_test_ds.X + 0.1 * np.random.randn(*noisy_test_ds.X.shape)
    from torch.utils.data import DataLoader
    noisy_test_loader = DataLoader(noisy_test_ds, batch_size=256, shuffle=False,
                                   num_workers=4, pin_memory=True)

    print("=" * 50 + "\nExperiment 3.4: Domain-Specific Augmentation\n" + "=" * 50)

    # Train baseline (no augmentation)
    print("\n>>> Baseline model ...")
    base_model = ResNet1D()
    trainer_base = ECGTrainer(base_model, clean_train_loader, clean_test_loader, device,
                              'results/models/Phase3_4_Base')
    trainer_base.fit(epochs=10)

    # Train augmented model
    print("\n>>> Augmented model ...")
    aug_model = ResNet1D()
    trainer_aug = ECGTrainer(aug_model, aug_train_loader, clean_test_loader, device,
                             'results/models/Phase3_4_Aug')
    trainer_aug.fit(epochs=10)

    # Evaluate
    base_clean = trainer_base.evaluate(clean_test_loader)['f1']
    base_noisy = trainer_base.evaluate(noisy_test_loader)['f1']
    aug_clean = trainer_aug.evaluate(clean_test_loader)['f1']
    aug_noisy = trainer_aug.evaluate(noisy_test_loader)['f1']

    results = {
        "Baseline_Clean_F1": base_clean,
        "Baseline_Noisy_F1": base_noisy,
        "Baseline_Drop": base_clean - base_noisy,
        "Augmented_Clean_F1": aug_clean,
        "Augmented_Noisy_F1": aug_noisy,
        "Augmented_Drop": aug_clean - aug_noisy,
    }

    print(f"\n[Robustness] Baseline drop: {results['Baseline_Drop']:.4f}, "
          f"Augmented drop: {results['Augmented_Drop']:.4f}")

    save_json(results, 'results/logs/phase3_4_augmentation.json')
    print("\nPhase 3.4 done.")


if __name__ == "__main__":
    main()
