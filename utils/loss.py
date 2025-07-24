import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.encode import reconstruct_from_spikes

class SpikePositionLoss(nn.Module):
    def __init__(
        self,
        tau: float = 5.0,
        lambda_pos: float = 1.0,
        lambda_vr:  float = 0.01,
        gamma_stft: float = 1.0,  # Temporal focus factor
        reduction: str = "mean",
        r_target:    float = None,
        
        device: torch.device = None
    ):
        super().__init__()
        self.tau       = tau
        self.lambda_vr = lambda_vr
        self.lambda_pos = lambda_pos
        self.gamma_stft  = gamma_stft
        self.reduction = reduction
        self.r_target  = r_target
        self.device    = device or torch.device("cpu")

        # Van Rossum kernel
        L = int(6 * tau)
        t_idx = torch.arange(0, L, device=self.device)
        kernel = torch.exp(-t_idx / tau).to(torch.float32)
        self.register_buffer("vr_kernel", kernel.view(1, 1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred, target: [B, T, C]
        mask: [B, T]
        """
        B, T, C = pred.shape

        # ====== Position Loss ======
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand(-1, -1, C)  # [B, T, C]
            pred_masked = pred[mask_exp == 1]                # [N_valid]
            target_masked = target[mask_exp == 1]            # [N_valid]
            pos_loss = F.binary_cross_entropy(pred_masked, target_masked.float())
        else:
            pos_loss = F.binary_cross_entropy(pred, target.float())

        # ====== Van Rossum Smoothing Loss ======
        p_in = pred.permute(0, 2, 1).reshape(B * C, 1, T)
        t_in = target.permute(0, 2, 1).reshape(B * C, 1, T)
        pad = self.vr_kernel.size(-1) // 2
        p_f = F.conv1d(p_in, self.vr_kernel, padding=pad)[..., :T]
        t_f = F.conv1d(t_in, self.vr_kernel, padding=pad)[..., :T]
        p_f = p_f.reshape(B, C, T).permute(0, 2, 1)
        t_f = t_f.reshape(B, C, T).permute(0, 2, 1)

        if mask is not None:
            vr_mask = mask.unsqueeze(-1).expand(-1, -1, C)  # [B, T, C]
            vr_loss = F.mse_loss(p_f[vr_mask == 1], t_f[vr_mask == 1])
        else:
            vr_loss = F.mse_loss(p_f, t_f)


        # ====== STFT Loss ======

        T_real = int(mask[0].sum().item())
        trimmed_spike_out = pred[0][:T_real]
        trimmed_target_spikes = target[0][:T_real]

        pred_reconstructed = reconstruct_from_spikes(trimmed_spike_out, mode='phased_rate', trim=True)
        target_reconstructed = reconstruct_from_spikes(trimmed_target_spikes, mode='phased_rate', trim=True)



        # 🎯 2. Time-Frequency Domain Loss (Magnitude Only)
        stft_loss = F.mse_loss(pred_reconstructed, target_reconstructed, reduction=self.reduction)
     

        # ====== Total Loss ======
        loss = self.lambda_pos * pos_loss + self.lambda_vr * vr_loss + self.gamma_stft * stft_loss
        print(f"Position Loss: {self.lambda_pos * pos_loss:.4f}, VR Loss: {self.lambda_vr * vr_loss:.4f}, STFT Loss: {self.gamma_stft * stft_loss:.4f}, Total Loss: {loss.item():.4f}")
        return loss


class FilterLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self,
                predicted_mask: torch.Tensor,        # [B, T, F]
                noisy_logstft: torch.Tensor,         # [B, T, F]
                clean_logstft: torch.Tensor,         # [B, T, F]
                mask: torch.Tensor = None            # [B, T] (optional)
                ) -> torch.Tensor:
        
        predicted_mask = predicted_mask.clamp(0.0, 1.0)
        denoised = predicted_mask * noisy_logstft  # [B, T, F]

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand_as(denoised)  # [B, T, F]
            denoised = denoised[mask_exp == 1]
            clean_logstft = clean_logstft[mask_exp == 1]

        loss = F.mse_loss(denoised, clean_logstft, reduction=self.reduction)
        print(f"[FilterLoss] MSE Loss: {loss.item():.4f}")
        return loss