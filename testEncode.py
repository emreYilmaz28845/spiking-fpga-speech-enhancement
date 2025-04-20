import os
import torch
import matplotlib.pyplot as plt
from encode import SpikeSpeechEnhancementDataset  # Adjust to your filename

# === Init dataset ===
data_root = "C:/VSProjects/spiking-fpga-project/audio"
noisy_dir = os.path.join(data_root, "noisy")
clean_dir = os.path.join(data_root, "clean")

dataset = SpikeSpeechEnhancementDataset(
    noisy_dir=noisy_dir,
    clean_dir=clean_dir,
    delta_threshold=0.003  # threshold for spike sparsity
)

# === Pick one sample ===
spikes, clean, normed_logmel = dataset[0]  # spikes: [T, n_mels], clean: [T, n_mels], logmel: [T, n_mels]

# === Reconstruct log-mel from spikes (simulates Sigma neuron) ===
reconstructed = spikes.cumsum(dim=0)

# === Plot original and encoded representations ===
fig, axs = plt.subplots(1, 3, figsize=(22, 5), sharey=True)

# 0. Original normalized log-mel
axs[0].imshow(normed_logmel.T, aspect='auto', origin='lower', interpolation='nearest')
axs[0].set_title("Original Log-Mel (normalized)")

# 1. Spike Magnitude Spectrogram
axs[1].imshow(spikes.abs().T, aspect='auto', origin='lower', interpolation='nearest')
axs[1].set_title("Spike Magnitude (|delta|)")

# 2. Reconstructed Log-Mel
axs[2].imshow(reconstructed.T, aspect='auto', origin='lower', interpolation='nearest')
axs[2].set_title("Reconstructed Log-Mel (∑spikes)")


# === Axis labels ===
for ax in axs:
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mel Bin")

plt.tight_layout()
plt.show()
