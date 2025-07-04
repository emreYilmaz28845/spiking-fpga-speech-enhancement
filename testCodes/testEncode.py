import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

from utils.encode import spike_encode, reconstruct_from_spikes

# === Config ===
wav_path = "audio/clean_16000/015.wav"

encoding_modes = ["delta", "rate", "phased_rate", "sod", "basic"]
encoding_thresholds = [0.003, 0.003, 0.003, 0.02, 0.5]
mse_results = {}

sample_rate = 16000
n_fft = 512
n_freq_bins = n_fft // 2 + 1
hop_length = 32
max_len = 5000
normalize_flag = True
padding = True

# === 1. Load audio
wave, sr = torchaudio.load(wav_path)
if sr != sample_rate:
    print(f"Resampling from {sr} to {sample_rate}...")
    wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wave)

# === 2. STFT transform (magnitude)
stft_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)
stft_mag = stft_transform(wave).squeeze(0)  # [n_freq, T]

# === 3. Precompute target STFT for comparison
_, normed_logstft, _, _, _ = spike_encode(
    stft_tensor=stft_mag,
    max_len=max_len,
    threshold=0.003,
    normalize=normalize_flag,
    mode="delta",
    padding=padding
)

# === Prepare output dirs
os.makedirs("testCodes/testFigures", exist_ok=True)

# === Figure A: Spike tensors
fig_spikes, axs_spikes = plt.subplots(len(encoding_modes), 1, figsize=(10, 2.5 * len(encoding_modes)), sharex=True)
if len(encoding_modes) == 1:
    axs_spikes = [axs_spikes]

# === Figure B: Reconstructed STFTs
fig_recons, axs_recons = plt.subplots(len(encoding_modes) + 1, 1, figsize=(10, 2.5 * (len(encoding_modes)+1)), sharex=True)

# Plot (0): Target STFT
axs_recons[0].imshow(normed_logstft.T.numpy(), aspect='auto', origin='lower', cmap='viridis')
axs_recons[0].set_title("Target Log-STFT")
axs_recons[0].set_ylabel("Freq Bin")

# === Loop through each mode
for i, mode in enumerate(encoding_modes):
    spikes, _, _, _, mask = spike_encode(
        stft_tensor=stft_mag,
        max_len=max_len,
        threshold=encoding_thresholds[i],
        normalize=normalize_flag,
        mode=mode,
        padding=padding
    )
    T_real = int(mask.sum().item())
    target_trimmed = normed_logstft[:T_real]
    reconstructed = reconstruct_from_spikes(spikes, mode=mode, mask=mask, trim=True)
    mse = torch.mean((target_trimmed - reconstructed) ** 2).item()
    mse_results[mode] = mse

    # === Plot spike tensor (delta özel işleme sahip)
    if mode == "delta":
        max_abs = spikes.abs().max().item()
        min_abs = spikes.abs().min().item()
        axs_spikes[i].imshow(
            spikes[:T_real].T.numpy(),
            aspect='auto',
            origin='lower',
            cmap='viridis',
            vmin=-min_abs,
            vmax=+max_abs
        )
    else:
        axs_spikes[i].imshow(
            spikes[:T_real].T.numpy(),
            aspect='auto',
            origin='lower',
            cmap='viridis',
            vmin=spikes[:T_real].min().item(),
            vmax=spikes[:T_real].max().item()
        )
    axs_spikes[i].set_title(f"{mode.upper()} Spikes")
    axs_spikes[i].set_ylabel("Freq Bin")

    # === Plot reconstruction
    axs_recons[i + 1].imshow(reconstructed.T.numpy(), aspect='auto', origin='lower', cmap='viridis')
    axs_recons[i + 1].set_title(f"{mode.upper()} Reconstructed")
    axs_recons[i + 1].set_ylabel("Freq Bin")


# Final labels
axs_spikes[-1].set_xlabel("Time")
axs_recons[-1].set_xlabel("Time")
fig_spikes.tight_layout()
fig_recons.tight_layout()

# Save figures
fig_spikes.savefig("testCodes/testFigures/spike_encode_spikes_only.png")
fig_recons.savefig("testCodes/testFigures/spike_encode_reconstructions_only.png")

# === MSE Bar Chart
plt.figure(figsize=(8, 4))
plt.bar(mse_results.keys(), mse_results.values())
plt.ylabel("MSE Loss")
plt.title("MSE of Reconstructed Log-STFT vs Target")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("testCodes/testFigures/spike_encode_comparison_chart_stft.png")

# === Print MSE values
for k, v in mse_results.items():
    print(f"[MSE] {k}: {v:.6f}")
