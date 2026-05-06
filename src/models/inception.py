import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    """单个 Inception 模块：bottleneck → 多尺度卷积 → concat + shortcut。"""

    def __init__(self, in_channels, out_channels=32, use_bottleneck=True, bottleneck_size=32):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        branch_ch = bottleneck_size if use_bottleneck else in_channels

        if use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, branch_ch, 1, bias=False)

        self.conv3 = nn.Conv1d(branch_ch, out_channels, 3, padding=1, bias=False)
        self.conv5 = nn.Conv1d(branch_ch, out_channels, 5, padding=2, bias=False)
        self.conv7 = nn.Conv1d(branch_ch, out_channels, 7, padding=3, bias=False)

        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        inp = self.bottleneck(x) if self.use_bottleneck else x
        b1 = self.conv3(inp)
        b2 = self.conv5(inp)
        b3 = self.conv7(inp)
        b4 = self.pool_conv(self.maxpool(x))

        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.act(self.bn(out))


class InceptionBlock(nn.Module):
    """带 residual shortcut 的 Inception 模块。"""

    def __init__(self, in_channels, out_channels=32, use_bottleneck=True, bottleneck_size=32):
        super().__init__()
        self.inception = InceptionModule(in_channels, out_channels, use_bottleneck, bottleneck_size)
        concat_ch = out_channels * 4

        self.shortcut = (nn.Conv1d(in_channels, concat_ch, 1, bias=False)
                         if in_channels != concat_ch else nn.Identity())
        self.bn_shortcut = (nn.BatchNorm1d(concat_ch)
                            if in_channels != concat_ch else nn.Identity())
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.inception(x)
        shortcut = self.bn_shortcut(self.shortcut(x))
        return self.act(out + shortcut)


class InceptionTime(nn.Module):
    """InceptionTime (Fawaz et al. 2019)。

    6 个 InceptionBlock + GAP + FC，每两个 block 之间有 residual shortcut。
    """

    def __init__(self, in_channels=1, num_classes=5, num_blocks=6,
                 out_channels=32, use_bottleneck=True, bottleneck_size=32):
        super().__init__()

        layers = []
        concat_ch = out_channels * 4
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else concat_ch
            layers.append(InceptionBlock(in_ch, out_channels, use_bottleneck, bottleneck_size))
        self.blocks = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(concat_ch, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
