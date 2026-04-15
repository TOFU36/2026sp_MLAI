import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.utils.tools import set_seed
from src.utils.logger import save_json
from src.data.dataset import ECGDataset
from src.models.networks import ResNet1D, BiLSTM, MLPMixer
from src.training.trainer import ECGTrainer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, device, input_shape=(1, 1, 187), num_trials=100):
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    # Warm up
    for _ in range(10): _ = model(dummy_input)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_trials):
            _ = model(dummy_input)
    # 返回每样本推理耗时 (ms)
    return ((time.time() - start_time) / num_trials) * 1000

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/raw/mitbih_train_downsampled_3000.csv', header=None).fillna(0)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:, -1], random_state=42)

    print("="*50 + "\n🧠 实验 3.2: 提取器归纳偏置之争 (1D-CNN vs LSTM vs MLP)\n" + "="*50)
    
    models_to_test = {
        "1D-ResNet": ResNet1D(kernel_size=7),
        "Bi-LSTM": BiLSTM(hidden_size=64),
        "MLP-Mixer": MLPMixer(seq_len=187)
    }

    train_loader = DataLoader(ECGDataset(train_df, feature_type='raw'), batch_size=64, shuffle=True)
    test_loader = DataLoader(ECGDataset(test_df, feature_type='raw'), batch_size=64, shuffle=False)
    
    results = {}
    for name, model in models_to_test.items():
        print(f"\n>>> Training {name}...")
        save_dir = f'results/models/Phase3_2_{name}'
        trainer = ECGTrainer(model, train_loader, test_loader, device, save_dir)
        history = trainer.fit(epochs=10)
        
        # 获取最终评估指标
        metrics = trainer.evaluate()
        params = count_parameters(model)
        latency = measure_latency(model, device)
        
        results[name] = {
            'F1': metrics['f1'],
            'AUPRC': metrics['auprc'],
            'Params': params,
            'Latency_ms': latency
        }
        print(f"-> {name}: F1={metrics['f1']:.4f}, Params={params}, Latency={latency:.2f}ms")

    save_json(results, 'results/logs/phase3_2_bias.json')
    print("\nPhase 3.2 实验完成，结果已保存！")

if __name__ == "__main__":
    main()