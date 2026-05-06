import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, make_loader
from src.models import ResNet1D, MLPMixer
from src.training.trainer import ECGTrainer
from src.features.transforms import ECGAugmentations


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()

    print("=" * 50)
    print("Experiment 3.5: Jitter Robustness (CNN vs MLP-Mixer vs RF)")
    print("=" * 50)

    # --- 加载 Phase 1 trial 0 的 CNN 模型（含池化层） ---
    cnn_ckpt = 'results/models/Phase1_ResNet_T0/best_model.pth'
    if not os.path.exists(cnn_ckpt):
        raise FileNotFoundError(f"{cnn_ckpt} not found — run Phase 1 first.")
    cnn_model = ResNet1D(kernel_size=7, use_se=False).to(device)
    cnn_model.load_state_dict(torch.load(cnn_ckpt, weights_only=True))
    print(f"Loaded CNN: {cnn_ckpt}")

    # --- 加载 Phase 3.2 trial 0 的 MLP-Mixer（无池化/卷积） ---
    mixer_ckpt = 'results/models/Phase3_2_MLP-Mixer_T0/best_model.pth'
    if not os.path.exists(mixer_ckpt):
        raise FileNotFoundError(f"{mixer_ckpt} not found — run Phase 3.2 first.")
    mixer_model = MLPMixer(seq_len=187, hidden_dim=256, num_layers=8,
                           tokens_ff_dim=256, channels_ff_dim=1024).to(device)
    mixer_model.load_state_dict(torch.load(mixer_ckpt, weights_only=True))
    print(f"Loaded MLP-Mixer: {mixer_ckpt}")

    # --- 训练 RF（ML baseline） ---
    print("\n>>> Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values.astype(int))

    # --- 评估：CNN 和 MLP-Mixer 通过 DataLoader，RF 直接用 numpy ---
    test_loader = make_loader(test_df)
    test_X = test_df.iloc[:, :-1].values
    test_y = test_df.iloc[:, -1].values.astype(int)

    cnn_trainer = ECGTrainer(cnn_model, test_loader, test_loader, device,
                             'results/models/Phase3_5_CNN')
    mixer_trainer = ECGTrainer(mixer_model, test_loader, test_loader, device,
                               'results/models/Phase3_5_Mixer')

    jitters = [0, 2, 5, 10, 15]
    cnn_scores, mixer_scores, rf_scores = [], [], []
    n_repeats = 5

    for j in jitters:
        print(f"\n>>> Jitter = +/-{j} steps")

        if j == 0:
            cnn_scores.append(cnn_trainer.evaluate(test_loader)['f1'])
            mixer_scores.append(mixer_trainer.evaluate(test_loader)['f1'])
            rf_scores.append(f1_score(test_y, rf.predict(test_X), average='macro'))
        else:
            cnn_f1s, mixer_f1s, rf_f1s = [], [], []
            for _ in range(n_repeats):
                jittered_X = np.array([ECGAugmentations.random_jitter(x, j) for x in test_X])

                # DL: 通过临时 DataLoader（jitter 已预施加到 df）
                jittered_df = test_df.copy()
                jittered_df.iloc[:, :-1] = jittered_X
                jitter_loader = make_loader(jittered_df, max_jitter=0)

                cnn_f1s.append(cnn_trainer.evaluate(jitter_loader)['f1'])
                mixer_f1s.append(mixer_trainer.evaluate(jitter_loader)['f1'])

                # ML: 直接用抖动后的 numpy 特征
                rf_f1s.append(f1_score(test_y, rf.predict(jittered_X), average='macro'))

            cnn_scores.append(float(np.mean(cnn_f1s)))
            mixer_scores.append(float(np.mean(mixer_f1s)))
            rf_scores.append(float(np.mean(rf_f1s)))

        print(f"    CNN: {cnn_scores[-1]:.4f}  |  Mixer: {mixer_scores[-1]:.4f}  |  RF: {rf_scores[-1]:.4f}")

    # 训练历史
    cnn_history = []
    phase1_json = 'results/logs/phase1_stats.json'
    if os.path.exists(phase1_json):
        with open(phase1_json) as f:
            cnn_history = json.load(f).get('dl_histories', [[]])[0]

    mixer_history = []
    phase32_json = 'results/logs/phase3_2_bias.json'
    if os.path.exists(phase32_json):
        with open(phase32_json) as f:
            p32 = json.load(f)
        mixer_histories = p32.get('histories', {}).get('MLP-Mixer', [])
        mixer_history = mixer_histories[0] if mixer_histories else []

    save_json({
        'jitters': jitters,
        'cnn_scores': [float(s) for s in cnn_scores],
        'mixer_scores': [float(s) for s in mixer_scores],
        'rf_scores': [float(s) for s in rf_scores],
        'cnn_history': cnn_history,
        'mixer_history': mixer_history,
    }, 'results/logs/phase3_5_jitter_robustness.json')
    print("\nPhase 3.5 done.")


if __name__ == "__main__":
    main()
