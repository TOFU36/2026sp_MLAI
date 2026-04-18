import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D
from src.training.trainer import ECGTrainer


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_df, test_df = load_train_test()
    train_loader, val_loader = create_dataloaders(train_df, test_df)

    # Task 2.1: Receptive Field (Kernel Size) Dynamics
    print("\n" + "=" * 50)
    print("Task 2.1: Receptive Field (Kernel Size) Dynamics")
    print("=" * 50)

    kernel_sizes = [3, 15, 31]
    kernel_results = {}

    for k in kernel_sizes:
        print(f"\n>>> Kernel Size = {k}")
        set_seed(42)
        model = ResNet1D(in_channels=1, num_classes=5, kernel_size=k, use_se=False)
        trainer = ECGTrainer(model, train_loader, val_loader, device,
                             f'results/models/Phase2_Kernel_{k}')
        history = trainer.fit(epochs=15, lr=1e-3, weight_decay=1e-4)
        kernel_results[f"Kernel_{k}"] = history

    save_json(kernel_results, 'results/logs/phase2_kernel_dynamics.json')

    # Task 2.2: Hyperparameter Grid Search (LR vs Weight Decay)
    print("\n" + "=" * 50)
    print("Task 2.2: LR vs Weight Decay Grid Search")
    print("=" * 50)

    learning_rates = [1e-2, 1e-3, 1e-4]
    weight_decays = [0.0, 1e-4, 1e-2]
    f1_matrix = np.zeros((len(learning_rates), len(weight_decays))).tolist()

    for i, lr in enumerate(learning_rates):
        for j, wd in enumerate(weight_decays):
            print(f"\n>>> LR={lr}, WD={wd}")
            set_seed(42)
            model = ResNet1D(in_channels=1, num_classes=5, kernel_size=15, use_se=False)
            trainer = ECGTrainer(model, train_loader, val_loader, device,
                                 f'results/models/Phase2_Grid_LR{lr}_WD{wd}')
            history = trainer.fit(epochs=10, lr=lr, weight_decay=wd)
            f1_matrix[i][j] = float(max(history['val_f1']))
            print(f"    Best F1-Macro = {f1_matrix[i][j]:.4f}")

    save_json({
        'learning_rates': learning_rates,
        'weight_decays': weight_decays,
        'f1_matrix': f1_matrix,
    }, 'results/logs/phase2_hyper_grid.json')
    print("\nPhase 2 done.")


if __name__ == "__main__":
    main()
