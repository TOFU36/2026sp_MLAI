import torch
import torch.nn as nn

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation 注意力模块 (Phase 4)"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResNet1D(nn.Module):
    """带注意力机制的 1D-ResNet 基线模型 (Phase 1 & 4)"""
    def __init__(self, in_channels=1, num_classes=5, kernel_size=7, use_se=True):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(64)
        
        # 残差投影
        self.shortcut = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64)
        )
        self.se = SEBlock1D(64) if use_se else nn.Identity()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def extract_features(self, x):
        """用于解耦分类器实验 (Phase 3.3)"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        res = self.shortcut(x)
        x = self.bn2(self.conv2(x))
        x += res
        x = self.relu(x)
        x = self.se(x)
        return self.global_pool(x).squeeze(-1) # 得到 Embedding

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc(features)

class InceptionTime(nn.Module):
    """SOTA架构: 多尺度并行特征提取 (Phase 5)"""
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, 32, 1, padding='same')
        self.conv10 = nn.Conv1d(32, 32, kernel_size=10, padding='same')
        self.conv20 = nn.Conv1d(32, 32, kernel_size=20, padding='same')
        self.conv40 = nn.Conv1d(32, 32, kernel_size=40, padding='same')
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, 32, 1, padding='same')
        
        self.bn = nn.BatchNorm1d(32 * 4)
        self.act = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        b1 = self.bottleneck(x)
        out1 = self.conv10(b1)
        out2 = self.conv20(b1)
        out3 = self.conv40(b1)
        out4 = self.pool_conv(self.maxpool(x))
        
        # 拼接不同感受野的特征
        x_concat = torch.cat([out1, out2, out3, out4], dim=1)
        x_concat = self.act(self.bn(x_concat))
        features = self.global_pool(x_concat).squeeze(-1)
        return self.fc(features)