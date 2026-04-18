import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders, make_loader
from src.models import ResNet1D
from src.training.trainer import ECGTrainer


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()
    train_loader, test_loader = create_dataloaders(train_df, test_df)

    print("=" * 50 + "\nExperiment 3.5: R-Peak Misalignment Robustness\n" + "=" * 50)

    # Train CNN (use test_loader as val_loader so fit() can track metrics)
    print(">>> Training 1D-CNN ...")
    cnn_model = ResNet1D()
    trainer = ECGTrainer(cnn_model, train_loader, test_loader, device,
                         'results/models/Phase3_5_CNN')
    trainer.fit(epochs=8)

    # Train RF
    print(">>> Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values.astype(int))

    # Test across jitter levels
    jitters = [0, 2, 5, 10, 15]
    cnn_scores, ml_scores = [], []

    for j in jitters:
        print(f"\n>>> Jitter = +/-{j} steps")
        jitter_loader = make_loader(test_df, max_jitter=j)
        jitter_ds = jitter_loader.dataset

        cnn_scores.append(trainer.evaluate(jitter_loader)['f1'])
        ml_scores.append(f1_score(jitter_ds.y, rf.predict(jitter_ds.X), average='macro'))
        print(f"    CNN F1: {cnn_scores[-1]:.4f}  |  RF F1: {ml_scores[-1]:.4f}")

    save_json({'jitters': jitters, 'cnn_scores': cnn_scores, 'ml_scores': ml_scores},
              'results/logs/phase3_5_jitter_robustness.json')
    print("\nPhase 3.5 done.")


if __name__ == "__main__":
    main()
