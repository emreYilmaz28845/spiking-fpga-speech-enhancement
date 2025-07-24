import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.audio_utils import reconstruct_without_stretch  # Griffin-Lim tabanlı reconstruction

class STFTLogWaveformLoss(nn.Module):
    def __init__(self, lambda_mag=0.5, lambda_sc=1, lambda_wave=0, eps=1e-8, cfg=None):
        super().__init__()
        self.lambda_mag = lambda_mag
        self.lambda_sc = lambda_sc
        self.lambda_wave = lambda_wave
        self.eps = eps
        self.cfg = cfg  # Griffin-Lim parametreleri için

    def forward(self, pred_log, target_log, mask=None, log_min=None, log_max=None, original_len=None):
        """
        pred_log, target_log: (B, F, T)
        mask: (B, T)
        log_min, log_max: (B,) tensor or float
        original_len: (B,) tensor
        """
        # === STFT domain loss ===
        if mask is not None:
            mask = mask.unsqueeze(1).to(pred_log.device)  # (B, 1, T)
            l1_diff = F.l1_loss(pred_log, target_log, reduction="none")
            mag_loss = (l1_diff * mask).sum() / (mask.sum() + self.eps)

            pred_mag = pred_log.exp() + self.eps
            target_mag = target_log.exp() + self.eps
            diff = (pred_mag - target_mag) ** 2

            sc_numer = torch.sum(diff * mask)
            sc_denom = torch.sum(target_mag ** 2 * mask) + self.eps
            sc_loss = sc_numer / sc_denom
        else:
            mag_loss = F.l1_loss(pred_log, target_log)
            pred_mag = pred_log.exp() + self.eps
            target_mag = target_log.exp() + self.eps
            sc_loss = (pred_mag - target_mag).norm(p='fro') / (target_mag.norm(p='fro') + self.eps)

        total_loss = self.lambda_mag * mag_loss + self.lambda_sc * sc_loss

        # === Griffin-Lim waveform-domain loss (optional) ===
        if self.lambda_wave > 0 and log_min is not None and log_max is not None and original_len is not None:
            waveform_loss = 0.0
            B = pred_log.size(0)
            for i in range(B):
                pred_wave = reconstruct_without_stretch(
                    pred_log[i].detach().cpu(), 
                    log_min[i].item(), log_max[i].item(),
                    n_fft=self.cfg.n_fft,
                    hop_length=self.cfg.hop_length,
                    sample_rate=self.cfg.sample_rate,
                    n_iter=self.cfg.n_iter,
                    original_length=original_len[i].item()
                )
                target_wave = reconstruct_without_stretch(
                    target_log[i].detach().cpu(), 
                    log_min[i].item(), log_max[i].item(),
                    n_fft=self.cfg.n_fft,
                    hop_length=self.cfg.hop_length,
                    sample_rate=self.cfg.sample_rate,
                    n_iter=self.cfg.n_iter,
                    original_length=original_len[i].item()
                )
                waveform_loss += F.l1_loss(pred_wave.to(pred_log.device), target_wave.to(pred_log.device))
            waveform_loss /= B
            total_loss += self.lambda_wave * waveform_loss
        print(f"Total Loss: {total_loss.item()}, Mag Loss: {mag_loss.item()}, SC Loss: {sc_loss.item()}")
        return total_loss


class STFTMagLoss(nn.Module):
    """
    Multi‑resolution STFT loss (L1 + spectral convergence) + opsiyonel time‑domain L1.
    Hızlı, autograd‑dostu.
    """
    def __init__(self, cfg, fft_sizes=None,
                 hop_ratio=0.25,
                 lambda_mag=0.3, lambda_sc=1.0, lambda_wave=0.0):
        super().__init__()
        self.cfg = cfg
        self.hop_ratio = hop_ratio
        self.lambda_mag, self.lambda_sc, self.lambda_wave = lambda_mag, lambda_sc, lambda_wave

        if fft_sizes is None:
            self.fft_sizes = (cfg.n_fft, cfg.n_fft * 2)
        else:
            self.fft_sizes = fft_sizes

    def stft_feats(self, x, n_fft):
        hop = int(n_fft * self.hop_ratio)
        win = torch.hann_window(n_fft, device=x.device)
        X = torch.stft(x, n_fft, hop, window=win,
                       return_complex=True, center=True, pad_mode='reflect')
        mag = torch.log10(X.abs() + 1e-8)
        return mag

    def forward(self, pred_wave, target_wave):
        mag_loss = 0.0
        sc_loss  = 0.0
        for n_fft in self.fft_sizes:
            P = self.stft_feats(pred_wave, n_fft)
            T = self.stft_feats(target_wave, n_fft)
            mag_loss += F.l1_loss(P, T)
            sc_loss  += (P - T).pow(2).sum() / (T.pow(2).sum() + 1e-8)

        total = self.lambda_mag * mag_loss + self.lambda_sc * sc_loss
        if self.lambda_wave > 0:
            total += self.lambda_wave * F.l1_loss(pred_wave, target_wave)
        return total
