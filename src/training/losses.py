import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(y):
    """计算逆频率类别权重（sklearn balanced 风格）"""
    counts = np.bincount(y)
    n_classes = len(counts)
    return [len(y) / (n_classes * c) for c in counts]


class FocalLoss(nn.Module):
    """
    Focal Loss 应对极端类别不平衡
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # alpha 可以是列表或张量，表示各个类别的权重
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
            
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss) # 获取预测正确的概率 p_t
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss