import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D
from src.training.trainer import ECGTrainer
from src.training.losses import FocalLoss, compute_class_weights
from src.evaluation.interpretability import GradCAM1D


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()
    y_train_all = train_df.iloc[:, -1].values.astype(int)
    alpha_weights = compute_class_weights(y_train_all)

    print("=" * 50 + "\nPhase 4: SE Attention & Grad-CAM\n" + "=" * 50)

    # SE=False baseline: 复用 Phase 1 trial 0 结果
    phase1_json = 'results/logs/phase1_stats.json'
    if os.path.exists(phase1_json):
        with open(phase1_json) as f:
            phase1_data = json.load(f)
        no_se_f1 = phase1_data['dl_test_f1'][0]
        no_se_history = phase1_data['dl_histories'][0]
        print(f"\n>>> SE=False baseline (reused from Phase 1): F1={no_se_f1:.4f}")
    else:
        raise FileNotFoundError(
            f"{phase1_json} not found — run Phase 1 first.")

    # Task 4.1: Train with SE attention（统一训练配置）
    print("\n>>> Training ResNet1D with SE-Attention ...")
    train_loader, val_loader, test_loader = create_dataloaders(train_df, test_df)
    model = ResNet1D(kernel_size=7, use_se=True).to(device)
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    trainer = ECGTrainer(model, train_loader, val_loader, device,
                         'results/models/Phase4_SE_ResNet', criterion=criterion)
    history = trainer.fit(epochs=100, lr=1e-3, weight_decay=1e-4)

    # Test set metrics
    test_metrics = trainer.evaluate(test_loader)
    se_f1 = test_metrics['f1']
    se_auprc = test_metrics['auprc']

    print(f"\n  SE=False F1: {no_se_f1:.4f}")
    print(f"  SE=True  F1: {se_f1:.4f}")

    # Task 4.2: Grad-CAM pathology atlas
    print("\n>>> Extracting Grad-CAM heatmaps ...")
    cam = GradCAM1D(model, model.target_layer)

    class_names = {
        1: 'Supraventricular ectopic (S)',
        2: 'Ventricular ectopic (V)',
        3: 'Fusion (F)',
        4: 'Unknown (Q)',
    }

    model.eval()
    atlas = {}
    for cls, name in class_names.items():
        row = test_df[test_df.iloc[:, -1] == cls].iloc[0]
        signal = row.iloc[:-1].values
        tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        heatmap = cam.generate_cam(tensor, target_class=cls)

        atlas[f"Class_{cls}"] = {'name': name, 'signal': signal.tolist(), 'heatmap': heatmap.tolist()}
        print(f"    -> {name}")

    save_json({
        'no_se_f1': no_se_f1,
        'se_f1': se_f1,
        'se_auprc': se_auprc,
        'no_se_history': no_se_history,
        'se_history': history,
        'pathology_atlas': atlas,
    }, 'results/logs/phase4_se_gradcam.json')
    print("\nPhase 4 done.")


if __name__ == "__main__":
    main()
