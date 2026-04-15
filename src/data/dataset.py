import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.features.transforms import ECGAugmentations, ModalityTransforms

class ECGDataset(Dataset):
    def __init__(self, df, feature_type='raw', augment=False, max_jitter=0):
        """
        feature_type: 'raw', 'fft', 'mel'
        augment: 是否进行数据增强 (Phase 3.4)
        max_jitter: 评估平移鲁棒性时的固定偏移量 (Phase 3.5)
        """
        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values.astype(int)
        self.feature_type = feature_type
        self.augment = augment
        self.max_jitter = max_jitter

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # 1. 评估时的特定偏移测试
        if self.max_jitter != 0:
            x = ECGAugmentations.random_jitter(x, self.max_jitter)
            
        # 2. 训练时的数据增强
        if self.augment:
            if np.random.rand() > 0.5:
                x = ECGAugmentations.add_baseline_drift(x)
            if np.random.rand() > 0.5:
                x = ECGAugmentations.add_gaussian_noise(x)
            if np.random.rand() > 0.5:
                x = ECGAugmentations.random_jitter(x, max_shift=5)

        # 3. 模态转换
        if self.feature_type == 'fft':
            x = ModalityTransforms.to_fft(x)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0) # (1, F)
        elif self.feature_type == 'mel':
            x = ModalityTransforms.to_mel_spectrogram(x)
            x = torch.tensor(x, dtype=torch.float32) # (1, M, T)
        else:
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0) # (1, 187)

        return x, torch.tensor(y, dtype=torch.long)