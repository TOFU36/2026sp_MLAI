import torch
import torch.nn as nn


class InceptionTime(nn.Module):
    """SOTA architecture: parallel multi-scale feature extraction (Phase 5)."""

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

        x_concat = torch.cat([out1, out2, out3, out4], dim=1)
        x_concat = self.act(self.bn(x_concat))
        features = self.global_pool(x_concat).squeeze(-1)
        return self.fc(features)
