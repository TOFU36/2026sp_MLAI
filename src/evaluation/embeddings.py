import numpy as np
import torch


def get_embeddings(model, loader, device):
    """Extract penultimate layer embeddings from a model."""
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            features = model.extract_features(X_batch.to(device))
            embs.append(features.cpu().numpy())
            labels.append(y_batch.numpy())
    return np.concatenate(embs), np.concatenate(labels)
