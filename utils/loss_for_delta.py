import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikePositionLossDelta(nn.Module):
    def __init__(
        self,
        tau: float = 1.0,
        lambda_pos: float = 100.0,
        lambda_vr: float = 0.1,
        lambda_cov: float = 5.0,            # ⬅ coverage kaybı ağırlığı
        min_coverage: float = 0.3,          # ⬅ minimum kabul edilen coverage
        device: torch.device = None
    ):
        super().__init__()
        self.tau = tau
        self.lambda_pos = lambda_pos
        self.lambda_vr = lambda_vr
        self.lambda_cov = lambda_cov
        self.min_coverage = min_coverage
        self.device = device or torch.device("cpu")

        # Precompute Van Rossum kernel
        L = int(6 * tau)
        t_idx = torch.arange(0, L, device=self.device)
        kernel = torch.exp(-t_idx / tau).to(torch.float32)
        self.register_buffer("vr_kernel", kernel.view(1, 1, -1))

    def log_cosh_loss(self, x, y):
        return torch.mean(torch.log(torch.cosh(x - y + 1e-12)))  # 1e-12 for numerical stability

    def compute_spike_coverage(self, pred, target, mask=None):
        eps = 1e-8
        pred_mask = (pred.abs() > 1e-4).float()
        target_mask = (target.abs() > 1e-4).float()

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand_as(target_mask)
            pred_mask *= mask_exp
            target_mask *= mask_exp

        overlap = (pred_mask * target_mask).sum()
        target_count = target_mask.sum()

        coverage = overlap / (target_count + eps)
        penalty = torch.relu(self.min_coverage - coverage) * self.lambda_cov
        return penalty, coverage.item()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred, target: [B, T, C] — real-valued delta-encoded spike outputs
        mask: [B, T] — optional padding mask
        """
        B, T, C = pred.shape

        # ====== Position-wise log-cosh loss ======
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand(-1, -1, C)  # [B, T, C]
            pos_loss = self.log_cosh_loss(pred[mask_exp == 1], target[mask_exp == 1])
        else:
            pos_loss = self.log_cosh_loss(pred, target)

        # ====== Van Rossum smoothing ======
        p_in = pred.permute(0, 2, 1).reshape(B * C, 1, T)
        t_in = target.permute(0, 2, 1).reshape(B * C, 1, T)
        pad = self.vr_kernel.size(-1) // 2
        p_f = F.conv1d(p_in, self.vr_kernel, padding=pad)[..., :T]
        t_f = F.conv1d(t_in, self.vr_kernel, padding=pad)[..., :T]
        p_f = p_f.reshape(B, C, T).permute(0, 2, 1)
        t_f = t_f.reshape(B, C, T).permute(0, 2, 1)

        if mask is not None:
            vr_loss = F.mse_loss(p_f[mask_exp == 1], t_f[mask_exp == 1])
        else:
            vr_loss = F.mse_loss(p_f, t_f)

        # ====== Spike Coverage Penalty ======
        coverage_penalty, coverage_value = self.compute_spike_coverage(pred, target, mask)

        # Optional: log coverage value for monitoring (not necessary for loss computation)
        self.last_coverage = coverage_value

        return self.lambda_pos * pos_loss + self.lambda_vr * vr_loss + coverage_penalty
