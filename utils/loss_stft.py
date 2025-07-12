import torch
import torch.nn as nn
import torch.nn.functional as F

# class STFTLogLoss(nn.Module):
#     def __init__(self, lambda_mag=1.0, lambda_sc=1.0):
#         super().__init__()
#         self.lambda_mag = lambda_mag
#         self.lambda_sc = lambda_sc

#     def forward(self, pred_log, target_log):
#         """
#         pred_log, target_log: [B, F, T] -- log-magnitude STFTs
#         """
#         pred_mag = torch.exp(pred_log) - 1e-6
#         target_mag = torch.exp(target_log) - 1e-6

#         mag_loss = F.l1_loss(pred_mag, target_mag)

#         sc_loss = (
#             torch.norm(target_mag - pred_mag, p='fro')
#             / (torch.norm(target_mag, p='fro') + 1e-8)
#         )

#         return self.lambda_mag * mag_loss + self.lambda_sc * sc_loss



class STFTLogLoss(nn.Module):
    def __init__(self, λ_mag=1.0, λ_sc=0.1, eps=1e-8):
        super().__init__()
        self.λ_mag, self.λ_sc, self.eps = λ_mag, λ_sc, eps

    def forward(self, pred_log, target_log):
        # 1) L1 kaybını log alanında tut
        mag_loss = F.l1_loss(pred_log, target_log)

        # 2) Spektral yakınsama (SC): lineer büyüklük + küçük ε
        pred_mag    = (pred_log  ).exp() + self.eps
        target_mag  = (target_log).exp() + self.eps
        sc_loss = (target_mag - pred_mag).norm(p='fro') / \
                  (target_mag.norm(p='fro') + self.eps)

        return self.λ_mag*mag_loss + self.λ_sc*sc_loss
