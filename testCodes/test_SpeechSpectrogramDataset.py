import sys
import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import soundfile as sf

# === Modül yolu ekle ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fast_cnn_dataloader import SpeechSpectrogramDataset
from utils.config import cfg

# === Dataset oluştur ===
dataset = SpeechSpectrogramDataset(cfg)
log_noisy, log_clean, log_min, log_max = dataset[0]
filename = dataset.filenames[0]

# === STFT Görselleştirme ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(log_noisy.T.numpy(), origin='lower', aspect='auto')
plt.title("Noisy log-STFT")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(log_clean.T.numpy(), origin='lower', aspect='auto')
plt.title("Clean log-STFT")
plt.colorbar()
plt.tight_layout()
plt.show()

# === Griffin-Lim ile log-STFT → waveform ===
def log_stft_to_waveform_griffinlim(log_mag, log_min, log_max, cfg):
    log_mag = log_mag.numpy()
    log_mag = log_mag * (log_max - log_min + 1e-8) + log_min
    mag = np.expm1(log_mag.T)  # (F, T)
    return librosa.griffinlim(
        mag,
        n_iter=cfg.n_iter,
        hop_length=cfg.hop_length,
        win_length=cfg.n_fft,
        window='hann',
        center=True,
        init='random',
        momentum=0.99
    )

# === Rekonstrüksiyon ===
noisy_wave = log_stft_to_waveform_griffinlim(log_noisy, log_min.item(), log_max.item(), cfg)
clean_wave = log_stft_to_waveform_griffinlim(log_clean, log_min.item(), log_max.item(), cfg)
sf.write("noisy_griffinlim.wav", noisy_wave, cfg.sample_rate)
sf.write("clean_griffinlim.wav", clean_wave, cfg.sample_rate)

# === Ground Truth waveform oku ===
clean_path = os.path.join(cfg.data_root, "clean", filename)
clean_gt_wave, _ = librosa.load(clean_path, sr=cfg.sample_rate)

# === Karşılaştırmak için truncate ===
min_len = min(len(clean_wave), len(clean_gt_wave))
clean_wave = clean_wave[:min_len]
clean_gt_wave = clean_gt_wave[:min_len]

# === Waveform Karşılaştırması ===
mse_wave = np.mean((clean_gt_wave - clean_wave) ** 2)
snr_wave = 10 * np.log10(np.sum(clean_gt_wave**2) / (np.sum((clean_gt_wave - clean_wave)**2) + 1e-8))
print(f"[Waveform] MSE: {mse_wave:.6f} | SNR: {snr_wave:.2f} dB")

plt.figure(figsize=(10, 3))
plt.plot(clean_gt_wave, label="Ground Truth", alpha=0.7)
plt.plot(clean_wave, label="Griffin-Lim Reconstructed", alpha=0.7)
plt.title("Waveform Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# === Ground Truth STFT (log1p abs) ===
stft_clean_gt = librosa.stft(clean_gt_wave, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
log_clean_gt = np.log1p(np.abs(stft_clean_gt))  # (F, T)

# === Dataset log_clean vs Ground Truth STFT ===
log_clean_ds = log_clean.T.numpy().T  # back to (F, T)
min_T = min(log_clean_gt.shape[1], log_clean_ds.shape[1])
log_clean_gt = log_clean_gt[:, :min_T]
log_clean_ds = log_clean_ds[:, :min_T]

mse_stft = np.mean((log_clean_gt - log_clean_ds) ** 2)
print(f"[Log-STFT] MSE: {mse_stft:.6f}")

# === STFT Karşılaştırma Görselleştirme ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(log_clean_ds, origin='lower', aspect='auto')
plt.title("Dataset Log-STFT")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(log_clean_gt, origin='lower', aspect='auto')
plt.title("GT Log-STFT (Recomputed)")
plt.colorbar()
plt.tight_layout()
plt.show()

# === Fark Spektrumu ===
diff_spec = np.abs(log_clean_ds - log_clean_gt)
plt.figure(figsize=(6, 4))
plt.imshow(diff_spec, origin='lower', aspect='auto', cmap='hot')
plt.title("|Dataset - Ground Truth| Log-STFT Diff")
plt.colorbar()
plt.tight_layout()
plt.show()
