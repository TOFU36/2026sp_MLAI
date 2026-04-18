import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, make_loader
from src.models import ResNet1D
from src.training.trainer import ECGTrainer


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()
    test_loader = make_loader(test_df)

    print("=" * 50 + "\nExperiment 3.5: R-Peak Misalignment Robustness\n" + "=" * 50)

    # 加载 Phase 1 trial 0 的模型（ResNet1D K=7, FocalLoss）
    ckpt_path = 'results/models/Phase1_ResNet_T0/best_model.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found — run Phase 1 first.")

    cnn_model = ResNet1D(kernel_size=7, use_se=False).to(device)
    cnn_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    trainer = ECGTrainer(cnn_model, test_loader, test_loader, device,
                         'results/models/Phase3_5_CNN')
    print(f"Loaded {ckpt_path}")

    # 训练 RF（ML baseline，用全量 train_df）
    print(">>> Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
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

    # 获取 Phase 1 trial 0 的训练历史
    phase1_json = 'results/logs/phase1_stats.json'
    cnn_history = []
    if os.path.exists(phase1_json):
        with open(phase1_json) as f:
            phase1_data = json.load(f)
        cnn_history = phase1_data.get('dl_histories', [[]])[0]

    save_json({'jitters': jitters, 'cnn_scores': cnn_scores, 'ml_scores': ml_scores,
               'cnn_history': cnn_history},
              'results/logs/phase3_5_jitter_robustness.json')
    print("\nPhase 3.5 done.")


if __name__ == "__main__":
    main()
