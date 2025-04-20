import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from spikerplus import NetBuilder

# === 1. Configuration ===
sample_path = "audio/noisy/001.wav"  # Replace with a valid file
model_path = "trained_models/trained_state_dict.pt"
sr = 16000
n_mels = 40
n_fft = 512
hop_length = 128
T_fixed = 1500  # Number of time steps used during training
threshold = 0.003  # Delta threshold (must match training)

# === 2. Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 3. Define SNN model ===
net_dict = {
    "n_cycles": T_fixed,
    "n_inputs": n_mels,
    "layer_0": {
        "neuron_model": "lif",
        "n_neurons": 128,
        "beta": 0.9375,
        "threshold": 0.1,
        "reset_mechanism": "subtract"
    },
    "layer_1": {
        "neuron_model": "lif",
        "n_neurons": 128,
        "beta": 0.9375,
        "threshold": 0.1,
        "reset_mechanism": "subtract"
    },
    "layer_2": {
        "neuron_model": "lif",
        "n_neurons": 40,
        "beta": 0.9375,
        "threshold": 0.1,
        "reset_mechanism": "none"
    }
}


snn = NetBuilder(net_dict).build(record_spikes=True, record_mem=True)

snn.load_state_dict(torch.load(model_path, map_location=device))
snn.to(device)
snn.eval()

# === 4. Preprocess input waveform ===
wav, sr = torchaudio.load(sample_path)
mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
mel = mel_transform(wav).squeeze(0)  # [n_mels, T]

# === 5. Delta Encoding (Same as training) ===
log_mel = torch.log(mel + 1e-6)
print(mel.min(), mel.max())
log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
log_mel_interp = torch.nn.functional.interpolate(log_mel.unsqueeze(0), size=T_fixed, mode="linear", align_corners=False).squeeze(0).T

input_spikes = torch.zeros_like(log_mel_interp)
prev = torch.zeros(log_mel_interp.shape[1])
for t in range(log_mel_interp.shape[0]):
    diff = log_mel_interp[t] - prev
    input_spikes[t] = (diff.abs() > threshold).float() * diff
    prev = log_mel_interp[t]

reconstructed_input = input_spikes.cumsum(dim=0)

spike_input = input_spikes.unsqueeze(1).to(device)  # [T, 1, F]

print("Input spikes stats:", input_spikes.abs().mean(), input_spikes.abs().max())
plt.hist(input_spikes.flatten().numpy(), bins=100)
plt.title("Input Spike Value Distribution")
plt.show()


# === 6. Inference ===
with torch.no_grad():
    # Run inference
    snn(spike_input)

    # === Inspect internal activity ===
    print("Layer 0 spike max:", snn.spk_rec['lif1'].max())
    print("Layer 1 spike max:", snn.spk_rec['lif2'].max())

    # === Final output membrane potential ===
    _, mem_out = list(snn.mem_rec.items())[-1]  # [T, 1, 40]
    output_spikes = mem_out.squeeze(1).cpu()    # [T, 40]

    print("Output spikes max:", output_spikes.abs().max())

    # === Reconstruct log-mel from output ===
    reconstructed_output = output_spikes.cumsum(dim=0).T  # [40, T]


# === 7. Visualization ===
fig, axs = plt.subplots(1, 5, figsize=(24, 5), sharey=True)

axs[0].imshow(log_mel.cpu(), aspect='auto', origin='lower')
axs[0].set_title("Input Normalized Log-Mel")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Mel Bin")

axs[1].imshow(input_spikes.abs().T, aspect='auto', origin='lower')
axs[1].set_title("Input Spike Magnitude |Δ|")
axs[1].set_xlabel("Time")

axs[2].imshow(reconstructed_input.T, aspect='auto', origin='lower')
axs[2].set_title("Input Spike ∑Δ (Reconstructed)")
axs[2].set_xlabel("Time")

axs[3].imshow(output_spikes.abs().T, aspect='auto', origin='lower')
axs[3].set_title("Output Spike Magnitude |Δ|")
axs[3].set_xlabel("Time")

axs[4].imshow(reconstructed_output, aspect='auto', origin='lower')
axs[4].set_title("Output Spike ∑Δ (Reconstructed)")
axs[4].set_xlabel("Time")

plt.tight_layout()
plt.show()
