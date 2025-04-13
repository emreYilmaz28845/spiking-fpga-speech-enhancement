import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from spikerplus import NetBuilder

# === 1. Load and preprocess waveform ===
waveform, sr = torchaudio.load("C:/VSProjects/spiking-fpga-project/audio/noisy/001.wav")  # Replace with your actual file
mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=512, hop_length=128, n_mels=40)
mel = mel_transform(waveform).squeeze(0)  # Shape: [40, T]

# === 2. Compress and normalize ===
mel = torch.pow(mel + 1e-6, 0.3)
mel = (mel - mel.mean()) / (mel.std() + 1e-6)

# === 3. Spike encoding via adaptive threshold ===
threshold = mel.mean() + 0.5 * mel.std()
spikes = (mel > threshold).float()  # Shape: [40, T]
spike_input = spikes.T.unsqueeze(0)  # Shape: [1, T, 40]

# === 4. Define Spiker+ SNN ===
net_dict = {
    "n_cycles": spike_input.shape[1],
    "n_inputs": spike_input.shape[2],
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

# === 5. Inference ===
# === 5. Inference ===
snn.eval()
with torch.no_grad():
    output_spikes = snn(spike_input)  # <- important fix


enhanced_mel = output_spikes.mean(dim=1).squeeze(0)  # [40]

# === 6. Reconstruct waveform using Griffin-Lim ===
enhanced_mel = enhanced_mel.unsqueeze(0)  # [1, 40]
inv_mel = T.InverseMelScale(n_stft=257, n_mels=40, sample_rate=sr)(enhanced_mel)
waveform_out = T.GriffinLim(n_fft=512, hop_length=128)(inv_mel)
torchaudio.save("C:/VSProjects/spiking-fpga-project/audio/enhanced/enhanced.wav", waveform_out.unsqueeze(0), sr)

# === 7. Visualize ===
plt.figure(figsize=(10, 4))
plt.title("Enhanced Mel Spectrogram")
plt.imshow(enhanced_mel.squeeze().cpu(), aspect='auto', origin='lower')
plt.colorbar()
plt.tight_layout()
plt.show()
