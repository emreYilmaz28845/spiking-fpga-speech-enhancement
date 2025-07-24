import torch
import os
from datetime import datetime
from utils.encode import reconstruct_from_spikes
from utils.audio_utils import reconstruct_without_stretch
from models.builder import build_network
from data.dataloader import get_dataloaders
from utils.config import cfg
from utils.plot_utils import plot_stft_comparison

# 1) load model
path = "Trained/2025-07-24_19-53_phased_rate_e5_len4000_arch_spiking-fsb-conv/trained_state_dict.pt"
snn = build_network(cfg)
snn.load_state_dict(torch.load(path))
snn.eval()

# 2) get one batch
_, val_loader = get_dataloaders(cfg)
(input_spikes,
 target_spikes,
 clean_logstft,
 noisy_logstft,
 log_min,
 log_max,
 original_length,
 mask) = next(iter(val_loader))

# [T, B, F] for inference
input_spikes = input_spikes.permute(1, 0, 2)

with torch.no_grad():
    # forward
    snn(input_spikes)
    _, rec = list(snn.spk_rec.items())[-1]     # [T, B, F]
    spikes = rec.permute(1, 0, 2)              # [B, T_padded, F]

    # true length
    T_real     = int(mask[0].sum().item())
    clean_spec = clean_logstft[0, :T_real, :].to(spikes.device)  # [T_real, F]
    noisy_spec = noisy_logstft[0, :T_real, :].to(spikes.device)  # [T_real, F]
    mask_trim  = mask[0, :T_real].to(spikes.device)              # [T_real]

    # clamp output to [0,1]
    pred_trim = spikes[0, :T_real, :].clamp(0.0,1.0)  # [T_real, F]

    # move to CPU and transpose to [F, T]
    clean_vis = clean_spec.cpu().T
    noisy_vis = noisy_spec.cpu().T

    if cfg.predict_filter:
        # 1) exp() → magnitude, 2) apply mask, 3) back to log‑STFT, so reconstructor sees log‑scale
        eps       = 1e-6
        mag_noisy = torch.exp(noisy_spec)                  # [T_real, F]
        mag_dn    = pred_trim * mag_noisy                  # [T_real, F]
        log_dn    = torch.log(mag_dn + eps)                # [T_real, F]
        pred_vis  = log_dn.cpu().T                         # [F, T_real]  ← now log‑STFT
        # ground truth is already log‑STFT:
        targ_vis  = clean_spec.cpu().T    
    else:
        # reconstruct log‐STFT out of spikes
        pred_vis = reconstruct_from_spikes(
            pred_trim, cfg.encode_mode, trim=True
        ).cpu().T
        targ_vis = reconstruct_from_spikes(
            target_spikes[0, :T_real], cfg.encode_mode, trim=True
        ).cpu().T

# build output folder
timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M")
suffix     = "masked" if cfg.predict_filter else "STFT"
out_folder = os.path.join("outputs", "wavs",
                          f"{timestamp}_{cfg.encode_mode}_len{cfg.max_len}_{suffix}")
os.makedirs(out_folder, exist_ok=True)

# write WAVs
# clean
reconstruct_without_stretch(
    clean_vis,
    log_min[0].item(), log_max[0].item(),
    os.path.join(out_folder,
                 "clean.wav" if cfg.predict_filter else "clean_STFT.wav"),
    n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    sample_rate=cfg.sample_rate, n_iter=cfg.n_iter
)

# noisy
reconstruct_without_stretch(
    noisy_vis,
    log_min[0].item(), log_max[0].item(),
    os.path.join(out_folder,
                 "noisy.wav" if cfg.predict_filter else "noisy_STFT.wav"),
    n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    sample_rate=cfg.sample_rate, n_iter=cfg.n_iter
)

# predicted
reconstruct_without_stretch(
    pred_vis,
    log_min[0].item(), log_max[0].item(),
    os.path.join(out_folder,
                 "predicted_masked.wav" if cfg.predict_filter else "predicted_STFT.wav"),
    n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    sample_rate=cfg.sample_rate, n_iter=cfg.n_iter
)

# target only in STFT‐mode
if not cfg.predict_filter:
    reconstruct_without_stretch(
        targ_vis,
        log_min[0].item(), log_max[0].item(),
        os.path.join(out_folder, "target_STFT.wav"),
        n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        sample_rate=cfg.sample_rate, n_iter=cfg.n_iter
    )

# final plot
plot_stft_comparison(
    out_folder,
    None if cfg.predict_filter else pred_trim.cpu().T,
    None if cfg.predict_filter else target_spikes[0, :T_real].cpu().T,
    pred_vis, targ_vis, clean_vis, snn
)

print(f"WAV files saved to: {out_folder}")
print(f"Plots saved to: {os.path.join(out_folder,'plots')}")
