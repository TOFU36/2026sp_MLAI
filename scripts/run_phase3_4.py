import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.utils.tools import set_seed
from src.utils.logger import save_json
from src.data.dataset import ECGDataset
from src.models.networks import ResNet1D
from src.training.trainer import ECGTrainer

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/raw/mitbih_train_downsampled_3000.csv', header=None).fillna(0)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:, -1], random_state=42)

    print("="*50 + "\n🛡️ 实验 3.4: 领域特定数据增强 (Robustness Drop Test)\n" + "="*50)

    # 训练集定义
    clean_train_loader = DataLoader(ECGDataset(train_df, augment=False), batch_size=64, shuffle=True)
    aug_train_loader = DataLoader(ECGDataset(train_df, augment=True), batch_size=64, shuffle=True)
    
    # 测试集定义
    clean_test_loader = DataLoader(ECGDataset(test_df, augment=False), batch_size=64, shuffle=False)
    # 为了测试鲁棒性，人为给测试集强制注入强烈噪声 (通过在数据集初始化时固定 augment=True 并在底层复用)
    # 这里我们使用一个定制化的噪音加载器
    noisy_test_dataset = ECGDataset(test_df, augment=False) 
    # 手动强制添加噪声以作纯粹的验证
    for i in range(len(noisy_test_dataset)):
        noisy_test_dataset.X[i] = noisy_test_dataset.X[i] + 0.1 * np.random.randn(187) # 添加高斯白噪声
    noisy_test_loader = DataLoader(noisy_test_dataset, batch_size=64, shuffle=False)

    results = {}
    
    # 1. 训练纯净模型
    print("\n>>> Training Baseline Model (No Augmentation)...")
    base_model = ResNet1D()
    trainer_base = ECGTrainer(base_model, clean_train_loader, clean_test_loader, device, 'results/models/Phase3_4_Base')
    trainer_base.fit(epochs=10)
    base_clean_f1 = trainer_base.evaluate(clean_test_loader)['f1']
    base_noisy_f1 = trainer_base.evaluate(noisy_test_loader)['f1']

    # 2. 训练带物理增强的模型
    print("\n>>> Training Augmented Model (Drift + Noise Jitter)...")
    aug_model = ResNet1D()
    trainer_aug = ECGTrainer(aug_model, aug_train_loader, clean_test_loader, device, 'results/models/Phase3_4_Aug')
    trainer_aug.fit(epochs=10)
    aug_clean_f1 = trainer_aug.evaluate(clean_test_loader)['f1']
    aug_noisy_f1 = trainer_aug.evaluate(noisy_test_loader)['f1']

    results = {
        "Baseline_Clean_F1": base_clean_f1,
        "Baseline_Noisy_F1": base_noisy_f1,
        "Baseline_Drop": base_clean_f1 - base_noisy_f1,
        "Augmented_Clean_F1": aug_clean_f1,
        "Augmented_Noisy_F1": aug_noisy_f1,
        "Augmented_Drop": aug_clean_f1 - aug_noisy_f1
    }
    
    print("\n[Robustness Report]:")
    print(f"Baseline Drop: {results['Baseline_Drop']:.4f}")
    print(f"Augmented Drop: {results['Augmented_Drop']:.4f}")
    
    save_json(results, 'results/logs/phase3_4_augmentation.json')
    print("\nPhase 3.4 实验完成，结果已保存！")

if __name__ == "__main__":
    main()