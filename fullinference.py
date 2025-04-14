import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from spikerplus import NetBuilder

# === 1. Configuration ===
sample_path = "audio/noisy/011.wav"  # Replace with a valid file
model_path = "trained_models/trained_state_dict.pt"
sr = 16000
n_mels = 40
n_fft = 512
hop_length = 128
T_fixed = 1500  # Number of time steps used during training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Define SNN model ===
net_dict = {
    "n_cycles": T_fixed,
    "n_inputs": n_mels,
    "layer_0": {
        "neuron_model": "lif",
        "n_neurons": 128,
        "beta": 0.9375,
        "threshold": 1.0,
        "reset_mechanism": "subtract"
    },
    "layer_1": {
        "neuron_model": "lif",
        "n_neurons": 40,
        "beta": 0.9375,
        "threshold": 1.0,
        "reset_mechanism": "none"
    }
}

snn = NetBuilder(net_dict).build()
snn.load_state_dict(torch.load(model_path, map_location=device))
snn.to(device)
snn.eval()

# === 3. Preprocess input waveform ===
waveform, sr = torchaudio.load(sample_path)
mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
mel = mel_transform(waveform).squeeze(0)  # [n_mels, T_orig]

# === 4. Rate coding ===
# Power-law compression + min-max normalization
mel = torch.pow(mel + 1e-6, 0.3)
mel = (mel - mel.min()) / (mel.max() - mel.min())

# Interpolate to fixed T and transpose
mel_interp = torch.nn.functional.interpolate(mel.unsqueeze(0), size=T_fixed, mode="linear", align_corners=False).squeeze(0)
spike_input = torch.bernoulli(mel_interp.T).unsqueeze(1).to(device)  # [T, 1, n_mels]

# === 5. Inference ===
with torch.no_grad():
    snn(spike_input)
    _, mem_out = list(snn.mem_rec.items())[-1]  # [T, 1, 40]
    enhanced_mel = mem_out.squeeze(1).cpu().T  # [40, T]

# === 6. Reconstruct waveform ===
inv_mel = T.InverseMelScale(n_stft=257, n_mels=40, sample_rate=sr)(enhanced_mel.unsqueeze(0))
waveform_out = T.GriffinLim(n_fft=n_fft, hop_length=hop_length)(inv_mel)
print("waveform_out shape:", waveform_out.shape)

torchaudio.save("audio/enhanced/enhanced.wav", waveform_out, sr)
torchaudio.save("audio/enhanced/noisy.wav", waveform, sr)




# === Plot original (noisy) and enhanced mel spectrograms ===
# === Plot original (noisy) and enhanced mel spectrograms ===
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# Noisy Mel Spectrogram
img0 = axs[0].imshow(mel.T.cpu(), aspect='auto', origin='lower')
axs[0].set_title("Noisy Mel Spectrogram")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Mel bins")
cbar0 = plt.colorbar(img0, ax=axs[0], orientation="vertical")
cbar0.set_label("Amplitude (normalized)")

# Enhanced Mel Spectrogram
img1 = axs[1].imshow(enhanced_mel, aspect='auto', origin='lower')
axs[1].set_title("Enhanced Mel Spectrogram")
axs[1].set_xlabel("Time")
cbar1 = plt.colorbar(img1, ax=axs[1], orientation="vertical")
cbar1.set_label("Amplitude (membrane potential)")

plt.tight_layout()
plt.show()


