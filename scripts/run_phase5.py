import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils.tools import set_seed
from src.utils.logger import save_json
from src.data.dataset import ECGDataset
from src.models.networks import ResNet1D, InceptionTime
from src.training.trainer import ECGTrainer

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 数据准备
    df = pd.read_csv('data/raw/mitbih_train_downsampled_3000.csv', header=None).fillna(0)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:, -1], random_state=42)
    
    train_loader = DataLoader(ECGDataset(train_df, feature_type='raw'), batch_size=64, shuffle=True)
    test_loader = DataLoader(ECGDataset(test_df, feature_type='raw'), batch_size=64, shuffle=False)

    print("="*50 + "\n🏆 Phase 5: SOTA 模型复现 (InceptionTime vs ResNet1D)\n" + "="*50)

    # 定义要终极比拼的架构
    models_to_test = {
        "ResNet1D (Baseline)": ResNet1D(kernel_size=7, use_se=True),
        "InceptionTime (SOTA)": InceptionTime() # 在 networks.py 中定义的并行多尺度卷积网络
    }

    results = {}
    
    for name, model in models_to_test.items():
        print(f"\n>>> Training {name}...")
        save_dir = f'results/models/Phase5_{name.split()[0]}'
        
        trainer = ECGTrainer(model, train_loader, test_loader, device, save_dir)
        # SOTA 对决，增加 epoch 以充分观察收敛极限
        history = trainer.fit(epochs=20, lr=1e-3, weight_decay=1e-4)
        
        metrics = trainer.evaluate()
        
        results[name] = {
            'train_loss': history['train_loss'],
            'val_f1_curve': history['val_f1'],
            'val_auprc_curve': history['val_auprc'],
            'final_f1': metrics['f1'],
            'final_auprc': metrics['auprc']
        }
        
        print(f"    -> Final Result for {name}: F1={metrics['f1']:.4f}, AUPRC={metrics['auprc']:.4f}")

    save_json(results, 'results/logs/phase5_sota_comparison.json')
    print("\nPhase 5 实验完成，SOTA 性能对决数据已保存至 logs/phase5_sota_comparison.json！")

if __name__ == "__main__":
    main()