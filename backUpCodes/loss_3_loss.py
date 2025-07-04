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
        gamma_focal: float = 2.0,   # focal loss gamma
        device: torch.device = None
    ):
        super().__init__()
        self.tau         = tau
        self.lambda_pos  = lambda_pos
        self.lambda_miss = lambda_miss
        self.lambda_vr   = lambda_vr
        self.lambda_freq = lambda_freq
        self.gamma_focal = gamma_focal
        self.device      = device or torch.device("cpu")

        # precompute Van Rossum kernel (time)
        L = int(6 * tau)
        t_idx = torch.arange(0, L, device=self.device, dtype=torch.float32)
        kernel = torch.exp(-t_idx / tau)
        self.register_buffer("vr_kernel", kernel.view(1, 1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
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

        # 2) Focal BCE for onsets
        w = 2.0  # reduce BCE weight scale
        W = 1.0 + w * mel
        # compute standard BCE per element
        bce_elem = F.binary_cross_entropy(pred, target.float(), weight=W, reduction='none')
        # focal factor: pts for positives, 1-pts for negatives
        pt = torch.where(target==1, pred, 1 - pred)
        focal_bce = ((1 - pt)**self.gamma_focal * bce_elem).mean()

        # 3) One-sided "miss" penalty
        misses = (target - pred).clamp(min=0.0)
        miss_pen = (misses * mel).mean()

        pos_loss = focal_bce + self.lambda_miss * miss_pen

        # 4) 2D frequency smoothing + Huber
        sigma = 1.0
        F_win = 7
        f_idx = torch.arange(F_win, device=pred.device) - (F_win//2)
        gauss = torch.exp(- (f_idx**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        freq_kernel = gauss.view(1, 1, 1, F_win)

        p2 = pred.unsqueeze(1)
        t2 = target.unsqueeze(1)
        pad_f = (F_win//2, F_win//2)
        p_smooth = F.conv2d(F.pad(p2, (pad_f[0], pad_f[1], 0, 0)), freq_kernel)
        t_smooth = F.conv2d(F.pad(t2, (pad_f[0], pad_f[1], 0, 0)), freq_kernel)
        freq_loss = F.smooth_l1_loss(p_smooth, t_smooth)

        # combine all terms
        loss = (
            self.lambda_pos  * pos_loss +
            self.lambda_vr   * vr_loss +
            self.lambda_freq * freq_loss
        )

        return loss
