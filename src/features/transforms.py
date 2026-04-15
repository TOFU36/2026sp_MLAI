import numpy as np
import torch
import torchaudio

class ECGAugmentations:
    @staticmethod
    def random_jitter(x, max_shift=10):
        """随机抖动模拟可穿戴设备R波未对齐 (Phase 3.5)"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(x, shift)
    
    @staticmethod
    def add_baseline_drift(x, fs=125):
        """模拟呼吸引起的低频基线漂移 (Phase 3.4)"""
        t = np.linspace(0, len(x)/fs, len(x))
        drift = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # 0.5Hz 正弦波
        return x + drift

    @staticmethod
    def add_gaussian_noise(x, snr=20):
        """添加高斯白噪声"""
        signal_power = np.mean(x**2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(x))
        return x + noise

class ModalityTransforms:
    @staticmethod
    def to_fft(x):
        """时域转频域幅值 (Phase 3.1)"""
        fft_out = torch.fft.rfft(torch.tensor(x))
        return torch.abs(fft_out).numpy()
    
    @staticmethod
    def to_mel_spectrogram(x, fs=125):
        """时域转梅尔频谱图 (2D 特征)"""
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=fs, n_fft=64, hop_length=16, n_mels=32
        )
        mel_spec = mel_transform(torch.tensor(x).float().unsqueeze(0))
        return mel_spec.numpy() # Shape: (1, n_mels, time)