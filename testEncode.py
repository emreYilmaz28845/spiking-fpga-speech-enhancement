import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from encode import spike_encode  # Your custom encoder

# === Config ===
wav_path = "audio/noisy/379.wav"  # 👈 set your test file here
sample_rate = 16000
n_mels = 40
n_fft = 512
hop_length = 128
max_len = 1500
threshold = 0.003

# === 1. Load audio
wave, sr = torchaudio.load(wav_path)
if sr != sample_rate:
    wave = torchaudio.transforms.Resample(sr, sample_rate)(wave)

# === 2. Mel-spectrogram
mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                 hop_length=hop_length, n_mels=n_mels)
mel = mel_transform(wave).squeeze(0)  # [n_mels, T]

# === 3. Encode
spikes, normed_logmel = spike_encode(
    mel_tensor=mel,
    max_len=max_len,
    threshold=threshold,
    normalize=True
)

# === 4. Reconstruct from spikes
reconstructed = spikes.cumsum(dim=0)

# === 5. Plot
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

axs[0].imshow(normed_logmel.T.numpy(), aspect='auto', origin='lower')
axs[0].set_title("Normalized Log-Mel")

axs[1].imshow(spikes.abs().T.numpy(), aspect='auto', origin='lower')
axs[1].set_title("Spike Magnitude (|delta|)")

axs[2].imshow(reconstructed.T.numpy(), aspect='auto', origin='lower')
axs[2].set_title("Reconstructed Log-Mel (∑spikes)")

for ax in axs:
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Bin")

plt.tight_layout()
plt.show()

# === 6. Optional MSE print
mse = torch.mean((normed_logmel - reconstructed) ** 2).item()
print(f"[INFO] MSE between original and reconstructed: {mse:.6f}")
