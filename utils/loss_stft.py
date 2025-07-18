import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.audio_utils import reconstruct_without_stretch #use this for waveform reconstruction


class STFTLogLoss(nn.Module):
    def __init__(self, λ_mag=1.0, λ_sc=0.5, eps=1e-8):
        super().__init__()
        self.λ_mag, self.λ_sc, self.eps = λ_mag, λ_sc, eps

    def forward(self, pred_log, target_log, mask=None):
        """
        pred_log, target_log: (B, F, T)
        mask: (B, T) or None
        """
        if mask is not None:
            # mask: (B, T) → (B, 1, T)
            mask = mask.unsqueeze(1).to(pred_log.device)

            # 1) L1 log-STFT loss (masked)
            l1_diff = F.l1_loss(pred_log, target_log, reduction="none")
            mag_loss = (l1_diff * mask).sum() / (mask.sum() + self.eps)

            # 2) Spectral Convergence (masked)
            pred_mag = pred_log.exp() + self.eps
            target_mag = target_log.exp() + self.eps
            diff = (target_mag - pred_mag) ** 2

            sc_numer = torch.sum(diff * mask)
            sc_denom = torch.sum(target_mag ** 2 * mask) + self.eps
            sc_loss = torch.sqrt(sc_numer / sc_denom)
        else:
            # Normal, masksiz versiyon
            mag_loss = F.l1_loss(pred_log, target_log)

            pred_mag = pred_log.exp() + self.eps
            target_mag = target_log.exp() + self.eps
            sc_loss = (target_mag - pred_mag).norm(p='fro') / \
                      (target_mag.norm(p='fro') + self.eps)

        return self.λ_mag * mag_loss + self.λ_sc * sc_loss
