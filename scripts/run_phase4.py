import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils.tools import set_seed
from src.utils.logger import save_json
from src.data.dataset import ECGDataset
from src.models.networks import ResNet1D
from src.training.trainer import ECGTrainer
from src.evaluation.interpretability import GradCAM1D

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 数据准备
    df = pd.read_csv('data/raw/mitbih_train_downsampled_3000.csv', header=None).fillna(0)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:, -1], random_state=42)
    
    train_loader = DataLoader(ECGDataset(train_df, feature_type='raw'), batch_size=64, shuffle=True)
    test_loader = DataLoader(ECGDataset(test_df, feature_type='raw'), batch_size=64, shuffle=False)

    print("="*50 + "\n👁️ Phase 4: 黑盒破局 - SE Attention & Grad-CAM\n" + "="*50)

    # 2. 训练带有 SE (Squeeze-and-Excitation) 注意力机制的 1D-ResNet (任务 4.1)
    print(">>> Training ResNet1D with SE-Attention Block...")
    model_se = ResNet1D(use_se=True).to(device)
    trainer = ECGTrainer(model_se, train_loader, test_loader, device, 'results/models/Phase4_SE_ResNet')
    trainer.fit(epochs=10) # 训练以获得有意义的权重
    
    # 3. 提取各类异常心跳的 Grad-CAM 热力图 (任务 4.2)
    print("\n>>> Extracting Grad-CAM Heatmaps for Pathological Atlas...")
    # 提取最后一次卷积层作为目标层 (在 networks.py 中定义为 conv2)
    cam_extractor = GradCAM1D(model_se, model_se.conv2)
    
    # 字典保存 4 种异常类的信号与热力图 (Class 1, 2, 3, 4)
    pathology_atlas = {}
    class_names = {1: 'Supraventricular ectopic (S)', 
                   2: 'Ventricular ectopic (V)', 
                   3: 'Fusion (F)', 
                   4: 'Unknown (Q)'}
                   
    model_se.eval()
    for class_idx in[1, 2, 3, 4]:
        # 从测试集中寻找该类别的第一个样本
        sample_row = test_df[test_df.iloc[:, -1] == class_idx].iloc[0]
        signal = sample_row.iloc[:-1].values
        
        # 转换为 tensor (batch, channel, seq_len)
        tensor_signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # 生成热力图
        cam_heatmap = cam_extractor.generate_cam(tensor_signal, target_class=class_idx)
        
        pathology_atlas[f"Class_{class_idx}"] = {
            'name': class_names[class_idx],
            'signal': signal.tolist(),
            'heatmap': cam_heatmap.tolist()
        }
        print(f"    -> Extracted Attention Heatmap for {class_names[class_idx]}")

    save_json(pathology_atlas, 'results/logs/phase4_pathology_atlas.json')
    print("\nPhase 4 实验完成，病理学图册数据已保存至 logs/phase4_pathology_atlas.json！")

if __name__ == "__main__":
    main()