import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D, InceptionTime
from src.training.trainer import ECGTrainer


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()
    train_loader, test_loader = create_dataloaders(train_df, test_df)

    print("=" * 50 + "\nPhase 5: InceptionTime vs ResNet1D\n" + "=" * 50)

    models = {
        "ResNet1D": ResNet1D(kernel_size=7, use_se=True),
        "InceptionTime": InceptionTime(),
    }

    results = {}
    for name, model in models.items():
        print(f"\n>>> {name}")
        trainer = ECGTrainer(model, train_loader, test_loader, device,
                             f'results/models/Phase5_{name}')
        history = trainer.fit(epochs=20, lr=1e-3, weight_decay=1e-4)
        metrics = trainer.evaluate()

        results[name] = {
            'train_loss': history['train_loss'],
            'val_f1_curve': history['val_f1'],
            'val_auprc_curve': history['val_auprc'],
            'final_f1': metrics['f1'],
            'final_auprc': metrics['auprc'],
        }
        print(f"  -> F1={metrics['f1']:.4f}, AUPRC={metrics['auprc']:.4f}")

    save_json(results, 'results/logs/phase5_sota_comparison.json')
    print("\nPhase 5 done.")


if __name__ == "__main__":
    main()
