import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D
from src.training.trainer import ECGTrainer
from src.evaluation.interpretability import GradCAM1D


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()
    train_loader, test_loader = create_dataloaders(train_df, test_df)

    print("=" * 50 + "\nPhase 4: SE Attention & Grad-CAM\n" + "=" * 50)

    # Task 4.1: Train with SE attention
    print(">>> Training ResNet1D with SE-Attention ...")
    model = ResNet1D(use_se=True).to(device)
    trainer = ECGTrainer(model, train_loader, test_loader, device,
                         'results/models/Phase4_SE_ResNet')
    trainer.fit(epochs=10)

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

    save_json(atlas, 'results/logs/phase4_pathology_atlas.json')
    print("\nPhase 4 done.")


if __name__ == "__main__":
    main()
