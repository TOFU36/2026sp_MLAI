import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.utils.tools import set_seed
from src.utils.logger import save_json
from src.data.dataset import ECGDataset
from src.models.networks import ResNet1D, ResNet2D
from src.training.trainer import ECGTrainer

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/raw/mitbih_train_downsampled_3000.csv', header=None).fillna(0)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:, -1], random_state=42)

    print("="*50 + "\n🔬 实验 3.1: 输入特征的模态之争\n" + "="*50)
    
    # 定义控制变量组：(输入类型, 使用的模型)
    experiments = {
        "1D_Raw": ('raw', ResNet1D(in_channels=1, kernel_size=7)),
        "1D_FFT": ('fft', ResNet1D(in_channels=1, kernel_size=7)), # FFT后依然是 1D 向量
        "2D_Mel": ('mel', ResNet2D(in_channels=1)),
        "2D_CWT": ('cwt', ResNet2D(in_channels=1))
    }

    results = {}
    for exp_name, (feat_type, model) in experiments.items():
        print(f"\n>>> Running Modality: {exp_name} (Feature: {feat_type})")
        train_loader = DataLoader(ECGDataset(train_df, feature_type=feat_type), batch_size=64, shuffle=True)
        test_loader = DataLoader(ECGDataset(test_df, feature_type=feat_type), batch_size=64, shuffle=False)
        
        save_dir = f'results/models/Phase3_1_{exp_name}'
        trainer = ECGTrainer(model, train_loader, test_loader, device, save_dir)
        history = trainer.fit(epochs=10) # 为了快速验证，设为10 epochs
        
        results[exp_name] = max(history['val_f1'])
        print(f"-> {exp_name} Best F1: {results[exp_name]:.4f}")

    save_json(results, 'results/logs/phase3_1_modality.json')
    print("\nPhase 3.1 实验完成，结果已保存！")

if __name__ == "__main__":
    main()