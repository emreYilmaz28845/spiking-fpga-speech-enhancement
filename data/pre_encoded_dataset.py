import os
import torch
import torchaudio
from utils.encode import spike_encode
from torchaudio.transforms import Spectrogram

def preprocess_and_save(noisy_dir, clean_dir, out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
    
    stft = Spectrogram(n_fft=cfg.n_fft, hop_length=cfg.hop_length, power=1.0)

    for i, filename in enumerate(noisy_files):
        name = os.path.splitext(filename)[0]

        noisy_wav, _ = torchaudio.load(os.path.join(noisy_dir, filename))
        clean_wav, _ = torchaudio.load(os.path.join(clean_dir, filename))

        noisy_spec = stft(noisy_wav).squeeze(0)
        clean_spec = stft(clean_wav).squeeze(0)

        noisy_spikes, noisy_normed, log_min, log_max, mask_noisy = spike_encode(
            stft_tensor=noisy_spec,
            max_len=cfg.max_len,
            threshold=cfg.threshold,
            normalize=cfg.normalize,
            mode=cfg.encode_mode,
            padding=cfg.padding,
        )
        clean_spikes, clean_normed, _, _, mask_clean = spike_encode(
            stft_tensor=clean_spec,
            max_len=cfg.max_len,
            threshold=cfg.threshold,
            normalize=cfg.normalize,
            mode=cfg.encode_mode,
            padding=cfg.padding,
        )

        data = {
            "noisy_spikes": noisy_spikes,
            "clean_spikes": clean_spikes,
            "noisy_normed": noisy_normed,
            "clean_normed": clean_normed,
            "log_min": log_min,
            "log_max": log_max,
            "mask": mask_clean,
            "orig_len": clean_wav.shape[-1]
        }

        torch.save(data, os.path.join(out_dir, f"{name}.pt"))

        if i % 50 == 0:
            print(f"[{i}/{len(noisy_files)}] Saved {name}.pt")
