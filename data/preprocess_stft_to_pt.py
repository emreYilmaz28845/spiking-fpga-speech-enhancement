# preprocess_stft_to_pt.py
import os
import librosa
import numpy as np
import torch
from tqdm import tqdm
from types import SimpleNamespace

cfg = SimpleNamespace(
    data_root="E:/VSProjects/datasets/audioVCTK",
    sample_rate=16000,
    n_fft=512,
    hop_length=256,
    max_len=800,
    normalize=True,
    encode_mode="delta"  # sadece klasör ismi için
)

# Çıktı dizinini oluştur
out_dir = f"{cfg.data_root}/preprocessed/{cfg.encode_mode}_Hop={cfg.hop_length}_Length={cfg.max_len}_NFFT={cfg.n_fft}_NoEncoding"
os.makedirs(out_dir, exist_ok=True)

clean_dir = os.path.join(cfg.data_root, "clean")
noisy_dir = os.path.join(cfg.data_root, "noisy")

filenames = sorted([
    f for f in os.listdir(clean_dir)
    if f.endswith(".wav") and os.path.exists(os.path.join(noisy_dir, f))
])

print(f"🚀 Processing {len(filenames)} files...")

for fname in tqdm(filenames):
    y_clean, _ = librosa.load(os.path.join(clean_dir, fname), sr=cfg.sample_rate)
    y_noisy, _ = librosa.load(os.path.join(noisy_dir, fname), sr=cfg.sample_rate)

    stft_clean = librosa.stft(y_clean, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    stft_noisy = librosa.stft(y_noisy, n_fft=cfg.n_fft, hop_length=cfg.hop_length)

    log_clean = np.log1p(np.abs(stft_clean))
    log_noisy = np.log1p(np.abs(stft_noisy))

    # Pad or truncate
    T = log_clean.shape[1]
    if T < cfg.max_len:
        pad_width = cfg.max_len - T
        log_clean = np.pad(log_clean, ((0, 0), (0, pad_width)), mode='constant')
        log_noisy = np.pad(log_noisy, ((0, 0), (0, pad_width)), mode='constant')
    elif T > cfg.max_len:
        log_clean = log_clean[:, :cfg.max_len]
        log_noisy = log_noisy[:, :cfg.max_len]

    # Transpose to (T, F)
    log_clean = log_clean.T
    log_noisy = log_noisy.T

    if cfg.normalize:
        log_min = min(log_clean.min(), log_noisy.min())
        log_max = max(log_clean.max(), log_noisy.max())
        log_clean = (log_clean - log_min) / (log_max - log_min + 1e-8)
        log_noisy = (log_noisy - log_min) / (log_max - log_min + 1e-8)
    else:
        log_min, log_max = 0.0, 1.0

    torch.save({
        "noisy_normed": torch.from_numpy(log_noisy).float(),
        "clean_normed": torch.from_numpy(log_clean).float(),
        "log_min": log_min,
        "log_max": log_max,
        "orig_len": min(T, cfg.max_len),
        "mask": torch.ones(cfg.max_len).float()  # sabit uzunluk
    }, os.path.join(out_dir, fname.replace(".wav", ".pt")))

print("✅ Done. Saved to:", out_dir)
