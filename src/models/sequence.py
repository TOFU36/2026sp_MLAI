import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """Bidirectional LSTM for time-series classification.

    Uses 2-layer BiLSTM (standard configuration) with hidden_size matching
    typical time-series literature.
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 1, T) → (B, T, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class MixerLayer(nn.Module):
    """One MLP-Mixer layer: token-mixing MLP + channel-mixing MLP."""

    def __init__(self, num_patches, hidden_dim, tokens_ff_dim, channels_ff_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_patches, tokens_ff_dim),
            nn.GELU(),
            nn.Linear(tokens_ff_dim, num_patches),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(hidden_dim, channels_ff_dim),
            nn.GELU(),
            nn.Linear(channels_ff_dim, hidden_dim),
        )

    def forward(self, x):
        # x: (B, num_patches, hidden_dim)
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mix(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.norm2(x)
        y = self.channel_mix(y)
        x = x + y
        return x


class MLPMixer(nn.Module):
    """MLP-Mixer (Tolstikhin et al. 2021) adapted for 1D time-series.

    Splits input signal into non-overlapping patches, then applies
    N MixerLayers with token-mixing and channel-mixing MLPs.
    """

    def __init__(self, seq_len=187, in_channels=1, patch_size=7,
                 hidden_dim=64, num_layers=4, tokens_ff_dim=128,
                 channels_ff_dim=256, num_classes=5):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(in_channels * patch_size, hidden_dim)

        self.mixer_layers = nn.Sequential(*[
            MixerLayer(self.num_patches, hidden_dim, tokens_ff_dim, channels_ff_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 187)
        x = x.squeeze(1)  # (B, 187)
        B, T = x.shape
        # Pad if not divisible
        patch_size = T // self.num_patches
        x = x[:, :self.num_patches * patch_size]
        x = x.reshape(B, self.num_patches, -1)  # (B, num_patches, patch_size)
        x = self.proj(x)  # (B, num_patches, hidden_dim)
        x = self.mixer_layers(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling over patches
        return self.head(x)
