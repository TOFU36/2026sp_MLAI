import time
import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """固定所有随机种子，保证实验的严格可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available compute device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(model, device, input_shape=(1, 1, 187), num_trials=100):
    """Measure per-sample inference latency in milliseconds."""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    for _ in range(10):
        _ = model(dummy_input)

    start = time.time()
    with torch.no_grad():
        for _ in range(num_trials):
            _ = model(dummy_input)
    return ((time.time() - start) / num_trials) * 1000