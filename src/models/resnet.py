import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel attention module."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)  # (B, C, 1)
        return x * w


# ---------------------------------------------------------------------------
# 1D ResNet
# ---------------------------------------------------------------------------

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 downsample=None, use_se=False):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1D(out_ch) if use_se else nn.Identity()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out, inplace=True)
        return self.se(out)


class ResNet1D(nn.Module):
    """Standard deep 1D-ResNet for ECG classification.

    Args:
        layers: block counts per stage. [2,2,2,2] = ResNet-18, [3,4,6,3] = ResNet-34.
        kernel_size: convolution kernel size used in ALL BasicBlocks.
        stem_kernel_size: kernel size for the initial stem convolution.
        use_se: insert SE attention into each BasicBlock.
        dropout: dropout rate before the final classifier.
    """

    def __init__(self, in_channels=1, num_classes=5, layers=None,
                 kernel_size=3, stem_kernel_size=15,
                 use_se=False, dropout=0.3):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]

        # Stem
        sp = stem_kernel_size // 2
        self.stem_conv = nn.Conv1d(in_channels, 64, stem_kernel_size,
                                   stride=2, padding=sp, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.in_ch = 64
        self.stage1 = self._make_stage(64, layers[0], kernel_size, stride=1, use_se=use_se)
        self.stage2 = self._make_stage(128, layers[1], kernel_size, stride=2, use_se=use_se)
        self.stage3 = self._make_stage(256, layers[2], kernel_size, stride=2, use_se=use_se)
        self.stage4 = self._make_stage(512, layers[3], kernel_size, stride=2, use_se=use_se)

        # Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, out_ch, num_blocks, kernel_size, stride, use_se):
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        blocks = [BasicBlock1D(self.in_ch, out_ch, kernel_size, stride, downsample, use_se)]
        self.in_ch = out_ch
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock1D(out_ch, out_ch, kernel_size, use_se=use_se))
        return nn.Sequential(*blocks)

    def extract_features(self, x):
        x = self.relu(self.bn1(self.stem_conv(x)))
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.global_pool(x).squeeze(-1)

    @property
    def target_layer(self):
        """Last conv layer for Grad-CAM."""
        return self.stage4[-1].conv2

    def forward(self, x):
        return self.fc(self.drop(self.extract_features(x)))


# ---------------------------------------------------------------------------
# 2D ResNet
# ---------------------------------------------------------------------------

class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size,
                               stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out, inplace=True)


class ResNet2D(nn.Module):
    """Standard deep 2D-ResNet for spectrogram / CWT inputs."""

    def __init__(self, in_channels=1, num_classes=5, layers=None,
                 kernel_size=3, dropout=0.3):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]

        self.stem_conv = nn.Conv2d(in_channels, 64, 7,
                                   stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_ch = 64
        self.stage1 = self._make_stage(64, layers[0], kernel_size, stride=1)
        self.stage2 = self._make_stage(128, layers[1], kernel_size, stride=2)
        self.stage3 = self._make_stage(256, layers[2], kernel_size, stride=2)
        self.stage4 = self._make_stage(512, layers[3], kernel_size, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * BasicBlock2D.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, out_ch, num_blocks, kernel_size, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        blocks = [BasicBlock2D(self.in_ch, out_ch, kernel_size, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock2D(out_ch, out_ch, kernel_size))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.relu(self.bn1(self.stem_conv(x)))
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).flatten(1)
        return self.fc(self.drop(x))
