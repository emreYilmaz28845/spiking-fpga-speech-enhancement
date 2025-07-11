import torch
import torch.nn as nn
import torch.nn.functional as F

class STFTLogLoss(nn.Module):
    def __init__(self, lambda_mag=1.0, lambda_sc=1.0):
        super().__init__()
        self.lambda_mag = lambda_mag
        self.lambda_sc = lambda_sc

    def forward(self, pred_log, target_log):
        """
        pred_log, target_log: [B, F, T] -- log-magnitude STFTs
        """
        pred_mag = torch.exp(pred_log) - 1e-6
        target_mag = torch.exp(target_log) - 1e-6

        mag_loss = F.l1_loss(pred_mag, target_mag)

        sc_loss = (
            torch.norm(target_mag - pred_mag, p='fro')
            / (torch.norm(target_mag, p='fro') + 1e-8)
        )

        return self.lambda_mag * mag_loss + self.lambda_sc * sc_loss
