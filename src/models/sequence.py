import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """Extracts global temporal memory via bidirectional LSTM."""

    def __init__(self, input_size=1, hidden_size=64, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 1, T) → (B, T, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class MLPMixer(nn.Module):
    """Pure MLP baseline with no sequential inductive bias."""

    def __init__(self, seq_len=187, num_classes=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))
