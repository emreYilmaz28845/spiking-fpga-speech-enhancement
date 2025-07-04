import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikePositionLoss(nn.Module):
    def __init__(
        self,
        tau: float = 5.0,
        lambda_pos: float = 1.0,    # weight of the position-wise term
        lambda_miss: float = 5.0,   # weight of the one-sided miss penalty
        lambda_vr: float = 0.01,    # weight of the Van-Rossum term
        lambda_freq: float = 1.0,   # weight of the 2D frequency-smoothing term
        device: torch.device = None
    ):
        super().__init__()
        self.tau         = tau
        self.lambda_pos  = lambda_pos
        self.lambda_miss = lambda_miss
        self.lambda_vr   = lambda_vr
        self.lambda_freq = lambda_freq
        self.device      = device or torch.device("cpu")

        # precompute Van Rossum kernel (time)
        L = int(6 * tau)
        t_idx = torch.arange(0, L, device=self.device, dtype=torch.float32)
        kernel = torch.exp(-t_idx / tau)
        # shape for conv1d: [out_ch=1, in_ch=1, kernel_size=L]
        self.register_buffer("vr_kernel", kernel.view(1, 1, -1))

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mel:    torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          pred, target: [B, T, F], in {0,1}
          mel          : [B, T, F], normalized log-Mel in [0,1]
        """
        B, T, C = pred.shape

        # 1) Van-Rossum smoothing loss (time)
        p_in = pred.permute(0,2,1).reshape(B*C, 1, T)
        t_in = target.permute(0,2,1).reshape(B*C, 1, T)
        pad_t = self.vr_kernel.size(-1) // 2
        p_f  = F.conv1d(p_in, self.vr_kernel, padding=pad_t)[..., :T]
        t_f  = F.conv1d(t_in, self.vr_kernel, padding=pad_t)[..., :T]
        p_f  = p_f.reshape(B, C, T).permute(0,2,1)
        t_f  = t_f.reshape(B, C, T).permute(0,2,1)
        vr_loss = F.mse_loss(p_f, t_f)
        print(f"Van-Rossum loss: {vr_loss.item():.6f} (lambda_vr * vr_loss = {self.lambda_vr * vr_loss.item():.6f})")

        # 2) Magnitude-weighted BCE
        w = 4.0
        W = 1.0 + w * mel
        bce = F.binary_cross_entropy(pred, target.float(), weight=W, reduction="mean")
        print(f"BCE loss: {bce.item():.6f}")

        # 3) One-sided "miss" penalty
        misses = (target - pred).clamp(min=0.0)
        miss_pen = (misses * mel).mean()
        print(f"Miss penalty: {miss_pen.item():.6f} (lambda_miss * miss_pen = {self.lambda_miss * miss_pen.item():.6f})")

        # combine position-wise terms
        pos_loss = bce + self.lambda_miss * miss_pen
        print(f"Position-wise loss (BCE + miss): {pos_loss.item():.6f} (lambda_pos * pos_loss = {self.lambda_pos * pos_loss.item():.6f})")

        # 4) 2D frequency smoothing + MSE
        sigma = 1.0
        F_win = 7
        f_idx = torch.arange(F_win, device=pred.device) - (F_win//2)
        gauss = torch.exp(- (f_idx**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        freq_kernel = gauss.view(1, 1, 1, F_win)

        p2 = pred.unsqueeze(1)   # [B,1,T,F]
        t2 = target.unsqueeze(1)
        pad_f = (F_win//2, F_win//2)
        p_smooth = F.conv2d(F.pad(p2, (pad_f[0], pad_f[1], 0, 0)), freq_kernel)
        t_smooth = F.conv2d(F.pad(t2, (pad_f[0], pad_f[1], 0, 0)), freq_kernel)
        freq_loss = F.mse_loss(p_smooth, t_smooth)
        print(f"Freq smoothing loss: {freq_loss.item():.6f} (lambda_freq * freq_loss = {self.lambda_freq * freq_loss.item():.6f})")

        # total
        loss = (
            self.lambda_pos  * pos_loss +
            self.lambda_vr   * vr_loss +
            self.lambda_freq * freq_loss
        )
        print(f"Total loss: {loss.item():.6f}\n")

        return loss
