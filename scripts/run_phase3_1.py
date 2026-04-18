import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D, ResNet2D
from src.training.trainer import ECGTrainer


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()

    print("=" * 50 + "\nExperiment 3.1: Input Modality Comparison\n" + "=" * 50)

    experiments = {
        "1D_Raw": ('raw', ResNet1D(in_channels=1, kernel_size=7)),
        "1D_FFT": ('fft', ResNet1D(in_channels=1, kernel_size=7)),
        "2D_Mel": ('mel', ResNet2D(in_channels=1)),
        "2D_CWT": ('cwt', ResNet2D(in_channels=1)),
    }

    results = {}
    for name, (feat_type, model) in experiments.items():
        print(f"\n>>> {name} (feature: {feat_type})")
        train_loader, test_loader = create_dataloaders(train_df, test_df, feature_type=feat_type)
        trainer = ECGTrainer(model, train_loader, test_loader, device,
                             f'results/models/Phase3_1_{name}')
        history = trainer.fit(epochs=10)
        results[name] = max(history['val_f1'])
        print(f"  -> Best F1: {results[name]:.4f}")

    save_json(results, 'results/logs/phase3_1_modality.json')
    print("\nPhase 3.1 done.")


if __name__ == "__main__":
    main()
