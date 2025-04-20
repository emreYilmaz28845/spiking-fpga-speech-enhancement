import torch
import torch.nn.functional as F

def spike_encode(mel_tensor, max_len=1500, threshold=0.003, normalize=True):
    mel = torch.log(mel_tensor + 1e-6)  # [n_mels, T]
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)  # normalize
    mel = F.interpolate(mel.unsqueeze(0), size=max_len, mode='linear', align_corners=False).squeeze(0).T  # [T, n_mels]

    deltas = torch.zeros_like(mel)
    prev = torch.zeros(mel.shape[1])
    for t in range(mel.shape[0]):
        diff = mel[t] - prev
        deltas[t] = (diff.abs() > threshold).float() * diff
        prev = mel[t]

    return deltas, mel  # [T, n_mels], normalized log-mel
