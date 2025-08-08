import torch
import os
from datetime import datetime
from utils.encode import reconstruct_from_spikes
from utils.audio_utils import reconstruct_without_stretch
from models.builder import build_network
from data.dataloader import get_dataloaders
from utils.config import cfg
from utils.plot_utils import plot_stft_comparison

# 1) Evaluate paths
paths = [
    "Trained/2025-08-01_12-52_phased_rate_e100_len4000_arch_spiking-fsb-conv/trained_state_dict.pt",
    "Trained/2025-06-04_01-18_phased_rate_e1_len10000/trained_state_dict.pt"
]

for path in paths:
    # Load model
    snn = build_network(cfg)
    snn.load_state_dict(torch.load(path))
    snn.eval()
    model_dir = os.path.dirname(path)
    out_folder = os.path.join("outputs", "wavs", os.path.basename(model_dir))
    os.makedirs(out_folder, exist_ok=True)

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

    input_spikes = input_spikes.permute(1, 0, 2)  # [T, B, F]

    with torch.no_grad():
        snn(input_spikes)
        _, rec = list(snn.spk_rec.items())[-1]
        spikes = rec.permute(1, 0, 2)  # [B, T_padded, F]

        T_real     = int(mask[0].sum().item())
        clean_spec = clean_logstft[0, :T_real, :].to(spikes.device)
        noisy_spec = noisy_logstft[0, :T_real, :].to(spikes.device)
        mask_trim  = mask[0, :T_real].to(spikes.device)

        pred_trim = spikes[0, :T_real, :].clamp(0.0, 1.0)

        clean_vis = clean_spec.cpu().T
        noisy_vis = noisy_spec.cpu().T

        if cfg.predict_filter:
            eps       = 1e-6
            mag_noisy = torch.exp(noisy_spec)
            mag_dn    = pred_trim * mag_noisy
            log_dn    = torch.log(mag_dn + eps)
            pred_vis  = log_dn.cpu().T
            targ_vis  = clean_spec.cpu().T
        else:
            pred_vis = reconstruct_from_spikes(pred_trim, cfg.encode_mode, trim=True).cpu().T
            targ_vis = reconstruct_from_spikes(target_spikes[0, :T_real], cfg.encode_mode, trim=True).cpu().T

    # Save WAVs
    reconstruct_without_stretch(
        clean_vis, log_min[0].item(), log_max[0].item(),
        os.path.join(out_folder, "clean.wav" if cfg.predict_filter else "clean_STFT.wav"),
        n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        sample_rate=cfg.sample_rate, n_iter=cfg.n_iter
    )
    reconstruct_without_stretch(
        noisy_vis, log_min[0].item(), log_max[0].item(),
        os.path.join(out_folder, "noisy.wav" if cfg.predict_filter else "noisy_STFT.wav"),
        n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        sample_rate=cfg.sample_rate, n_iter=cfg.n_iter
    )
    reconstruct_without_stretch(
        pred_vis, log_min[0].item(), log_max[0].item(),
        os.path.join(out_folder, "predicted_masked.wav" if cfg.predict_filter else "predicted_STFT.wav"),
        n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        sample_rate=cfg.sample_rate, n_iter=cfg.n_iter
    )
    if not cfg.predict_filter:
        reconstruct_without_stretch(
            targ_vis, log_min[0].item(), log_max[0].item(),
            os.path.join(out_folder, "target_STFT.wav"),
            n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            sample_rate=cfg.sample_rate, n_iter=cfg.n_iter
        )

    plot_stft_comparison(
        out_folder,
        None if cfg.predict_filter else pred_trim.cpu().T,
        None if cfg.predict_filter else target_spikes[0, :T_real].cpu().T,
        pred_vis, targ_vis, clean_vis, snn
    )

    print(f"✓ WAV files saved to: {out_folder}")
    print(f"✓ Plots saved to: {os.path.join(out_folder, 'plots')}")
