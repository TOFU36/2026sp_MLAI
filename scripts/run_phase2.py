import sys
import os
# 将项目根目录添加到系统路径
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 数据加载与预处理 (使用简单的 Train/Val 划分以节省探索性实验的时间)
    data_path = 'data/raw/mitbih_train_downsampled_3000.csv'
    try:
        df = pd.read_csv(data_path, header=None)
    except FileNotFoundError:
        print("未找到数据，生成虚拟数据用于代码测试流程...")
        X_dummy = np.random.randn(3000, 187) * 0.1
        y_dummy = np.random.choice([0, 1, 2, 3, 4], size=3000, p=[0.8, 0.05, 0.08, 0.02, 0.05])
        df = pd.DataFrame(X_dummy)
        df[187] = y_dummy

    X_df = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').fillna(0)
    df.iloc[:, :-1] = X_df
    
    # 划分出 20% 作为验证集
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:, -1], random_state=42)
    
    train_loader = DataLoader(ECGDataset(train_df, feature_type='raw'), batch_size=64, shuffle=True)
    val_loader = DataLoader(ECGDataset(val_df, feature_type='raw'), batch_size=64, shuffle=False)

    # =====================================================================
    # [Task 2.1]: 感受野 (Kernel Size) 与动力学耦合实验
    # 假设: 小核只能看噪音，大核能看清 QRS 宽波群的形态。
    # =====================================================================
    print("\n" + "="*50)
    print("🔬 Task 2.1: Receptive Field (Kernel Size) Dynamics")
    print("="*50)
    
    kernel_sizes = [3, 15, 31]
    kernel_results = {}
    
    for k in kernel_sizes:
        print(f"\n>>> Training ResNet1D with Kernel Size = {k}")
        set_seed(42) # 严格控制初始化的随机性变量
        model = ResNet1D(in_channels=1, num_classes=5, kernel_size=k, use_se=False)
        save_dir = f'results/models/Phase2_Kernel_{k}'
        
        # 使用标准的交叉熵即可，重点看收敛速度差异
        trainer = ECGTrainer(model, train_loader, val_loader, device, save_dir)
        
        # 训练 15 个 epoch，记录每一步的 loss 和 F1
        history = trainer.fit(epochs=15, lr=1e-3, weight_decay=1e-4)
        kernel_results[f"Kernel_{k}"] = history
        
    save_json(kernel_results, 'results/logs/phase2_kernel_dynamics.json')
    print("-> Task 2.1 感受野对比结果已保存至 results/logs/phase2_kernel_dynamics.json")

    # =====================================================================
    # [Task 2.2]: 优化空间的平滑性探讨 (LR & Weight Decay 网格搜索)
    # 目标: 绘制 2D 热力图探讨参数组合对模型抗过拟合（尤其针对少数类）的影响。
    # =====================================================================
    print("\n" + "="*50)
    print("🛠️ Task 2.2: Hyperparameter Grid Search (LR vs Weight Decay)")
    print("="*50)
    
    learning_rates = [1e-2, 1e-3, 1e-4]
    weight_decays = [0.0, 1e-4, 1e-2]
    
    grid_results = {
        'learning_rates': learning_rates,
        'weight_decays': weight_decays,
        'f1_matrix': np.zeros((len(learning_rates), len(weight_decays))).tolist()
    }
    
    # 选用表现最稳健的核大小(如 Kernel=15) 进行网格搜索
    for i, lr in enumerate(learning_rates):
        for j, wd in enumerate(weight_decays):
            print(f"\n>>> Grid Search: LR = {lr}, Weight Decay = {wd}")
            set_seed(42)
            model = ResNet1D(in_channels=1, num_classes=5, kernel_size=15, use_se=False)
            save_dir = f'results/models/Phase2_Grid_LR{lr}_WD{wd}'
            
            trainer = ECGTrainer(model, train_loader, val_loader, device, save_dir)
            # 为了加速网格搜索，仅训练 10 个 epoch
            history = trainer.fit(epochs=10, lr=lr, weight_decay=wd)
            
            # 取验证集上的最好 F1-macro 存入网格矩阵
            best_f1 = max(history['val_f1'])
            grid_results['f1_matrix'][i][j] = float(best_f1)
            print(f"    [Result]: Best F1-Macro = {best_f1:.4f}")

    save_json(grid_results, 'results/logs/phase2_hyper_grid.json')
    print("-> Task 2.2 网格搜索结果已保存至 results/logs/phase2_hyper_grid.json")
    print("\n✅ Phase 2 全部任务执行完毕！")

if __name__ == "__main__":
    main()