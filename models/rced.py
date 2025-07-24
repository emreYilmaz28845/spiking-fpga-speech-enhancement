import torch
import torch.nn as nn

class RCED10(nn.Module):
    def __init__(self, n_freq_bins=257, n_frames=8):  # n_freq_bins input channel!
        super().__init__()

        def block(in_ch, out_ch, k):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc = nn.Sequential(
            block(n_freq_bins, 12, 13),
            block(12, 16, 11),
            block(16, 20, 9),
            block(20, 24, 7),
            block(24, 32, 7),
        )

        self.dec = nn.Sequential(
            block(32, 24, 7),
            block(24, 20, 9),
            block(20, 16, 11),
            block(16, 12, 13),
            nn.Conv1d(12, n_freq_bins, kernel_size=1),  # No padding, match input channels
        )

    def forward(self, x):
        # x: [B, F, T] — already correct
        z = self.enc(x)
        y_hat = self.dec(z)
        return y_hat

