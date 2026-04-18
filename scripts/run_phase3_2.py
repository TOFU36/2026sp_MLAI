import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.tools import set_seed, get_device, count_parameters, measure_latency
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D, BiLSTM, MLPMixer
from src.training.trainer import ECGTrainer


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()
    train_loader, test_loader = create_dataloaders(train_df, test_df)

    print("=" * 50 + "\nExperiment 3.2: Extractor Inductive Bias\n" + "=" * 50)

    models_to_test = {
        "1D-ResNet": ResNet1D(kernel_size=7),
        "Bi-LSTM": BiLSTM(hidden_size=64),
        "MLP-Mixer": MLPMixer(seq_len=187),
    }

    results = {}
    for name, model in models_to_test.items():
        print(f"\n>>> {name}")
        trainer = ECGTrainer(model, train_loader, test_loader, device,
                             f'results/models/Phase3_2_{name}')
        trainer.fit(epochs=10)

        metrics = trainer.evaluate()
        params = count_parameters(model)
        latency = measure_latency(model, device)

        results[name] = {
            'F1': metrics['f1'],
            'AUPRC': metrics['auprc'],
            'Params': params,
            'Latency_ms': latency,
        }
        print(f"  -> F1={metrics['f1']:.4f}, Params={params}, Latency={latency:.2f}ms")

    save_json(results, 'results/logs/phase3_2_bias.json')
    print("\nPhase 3.2 done.")


if __name__ == "__main__":
    main()
