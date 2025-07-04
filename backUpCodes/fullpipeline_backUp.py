import os
import torch
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchaudio.functional import resample
import torchaudio
from torchaudio.transforms import GriffinLim
import sys
import os

import librosa
import soundfile as sf

# Add hifi-gan directory to Python path
sys.path.append(os.path.join("C:/VSProjects/spiking-fpga-project/hifi-gan/"))

from hifi_utils_module import load_hifigan_model, logmel_to_wav

from spikerplus import NetBuilder, Trainer
from spikerplus.vhdl import write_vhdl
from utils.encode import reconstruct_from_spikes
from data.dataset import SpikeSpeechEnhancementDataset

# Suppress debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)


def logstft_to_waveform(log_stft: torch.Tensor, n_fft: int = 1024, hop_length: int = 256,
                        win_length: int = None, sample_rate: int = 22050, n_iter: int = 32) -> torch.Tensor:
    """
    Converts log-magnitude STFT [F, T] to waveform using Griffin-Lim.

    Args:
        log_stft (Tensor): [F, T] log-magnitude spectrogram (e.g., output of spike reconstruction)
        n_fft (int): FFT size
        hop_length (int): hop length between frames
        win_length (int): window size (defaults to n_fft)
        sample_rate (int): for saving/listening if needed
        n_iter (int): Griffin-Lim iterations (more = better quality, slower)

    Returns:
        waveform (Tensor): [1, T] waveform (mono)
    """
    if win_length is None:
        win_length = n_fft

    # 1. Undo log scale
    magnitude = torch.exp(log_stft) - 1e-6  # [F, T]

    # 2. Griffin-Lim phase reconstruction
    griffinlim = GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=1.0,         # magnitude input (not power)
        n_iter=n_iter
    )
    
    # Input to GriffinLim should be [B=1, F, T]
    waveform = griffinlim(magnitude.unsqueeze(0))  # → [1, T]
    return waveform

# Parameters
sample_rate = 22050 #16000 #22050
n_fft = 1024
n_freq_bins = n_fft // 2 + 1
hop_length = 256
n_mels = 80
spike_threshold = 0.2
max_len = 1500
encode_mode = "phased_rate"  # "delta", "sod", "rate", "phased_rate", "basic"
normalize_flag = True  # Normalize the log-STFT

# Set encoding threshold by mode
if encode_mode == "sod":
    encode_threshold = 0.02
elif encode_mode == "delta":
    encode_threshold = 0.003
elif encode_mode == "basic":
    encode_threshold = 0.5
else:
    encode_threshold = 0.003

stats = torch.load("logmel_stats.pt")
logmel_min = stats["min"]
logmel_max = stats["max"]

# Load dataset
data_root = r"C:/VSProjects/spiking-fpga-project/audio"
dataset = SpikeSpeechEnhancementDataset(
    noisy_dir=os.path.join(data_root, "noisy_22050"),
    clean_dir=os.path.join(data_root, "clean_22050"),
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    threshold=encode_threshold,
    mode=encode_mode,
    max_len=max_len
)

# Split dataset
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False)

# Build network
example_input, _, _,_,_,_,_= dataset[0]
net_dict = {
    # -----------------------------------------------------------------
    "n_cycles" : example_input.shape[0],   # e.g. 1500
    "n_inputs" : example_input.shape[1],   # 128 Mel bins
    # -----------------   ENCODER  ------------------------------------
    # 128 ➜ 128   Syn (causal smoothing, learnable α/β)
    "layer_0": { "neuron_model": "syn",
                 "n_neurons": 128,
                 "alpha": 0.05,  "learn_alpha": False,
                 "beta":  0.05,  "learn_beta" : False,
                 "threshold": spike_threshold+0.4, "learn_threshold": False,
                 "reset_mechanism": "zero" },

    # 128 ➜ 256   RSyn (context memory ~50 ms)
    "layer_1": { "neuron_model": "rsyn",
                 "n_neurons": 256,
                 "alpha": 0.07,  "learn_alpha": False,
                 "beta" : 0.07,  "learn_beta" : False,
                 "threshold": spike_threshold+0.1, "learn_threshold": False,
                 "reset_mechanism": "zero", "bias": True },

    # -----------------  BOTTLENECK  ----------------------------------
    # 256 ➜  64   RIF (sparse latent speech code)
    "layer_2": { "neuron_model": "rif",
                 "n_neurons": 64,
                 "beta" : 0.5,  "learn_beta": False,  # RIF has no leak
                 "threshold": spike_threshold+0.1, "learn_threshold": False,
                 "reset_mechanism": "zero", "bias": True },

    # -----------------  DECODER  -------------------------------------
    # 64 (+ skip 128) ➜ 128   Syn (reconstruct)
    "layer_3": { "neuron_model": "syn",
                 "n_neurons": 128,
                 "alpha": 0.1,  "learn_alpha": False,
                 "beta" : 0.6,  "learn_beta" : False,
                 "threshold": spike_threshold, "learn_threshold": False,
                 "reset_mechanism": "subtract", "bias": True }, #can be use subtract?

    # 128 ➜ 128  IF  (spike read-out)
    "layer_4": { "neuron_model": "if",
                 "n_neurons": n_freq_bins,
                 "threshold": spike_threshold*0.3, "learn_threshold": False,
                 "reset_mechanism": "zero", "bias": False }
}

snn = NetBuilder(net_dict).build()
trainer = Trainer(snn)
#trainer.train(train_loader, val_loader, n_epochs=1, store=True, output_dir="trained_models")
#T = len(trainer.logs['target_rate'])
#batches = list(range(T))
#checkpoint = torch.load("trained_models/trained_state_dict.pt", map_location="cpu")
#snn.load_state_dict(checkpoint)
snn.eval()

# Run inference
sample_batch = next(iter(val_loader))
input_spikes, target_spikes, clean_logstft, noisy_logstft,log_min,log_max,original_length = sample_batch  # all [B, T, F]
input_spikes = input_spikes.permute(1, 0, 2)  # [T, B, F]

with torch.no_grad():
    snn(input_spikes)
    _, spike_out = list(snn.spk_rec.items())[-1]  # [T, B, F]
    spike_out = spike_out.permute(1, 0, 2)        # [B, T, F]
    
    # Only visualize first sample in batch
    pred_reconstructed = reconstruct_from_spikes(spike_out[0], mode=encode_mode)
    target_reconstructed = reconstruct_from_spikes(target_spikes[0], mode=encode_mode)

# Transpose to [F, T]
predicted_vis = pred_reconstructed.cpu().T
ground_truth_vis = target_reconstructed.cpu().T
pred_spikes = spike_out[0].cpu().T
target_spikes_vis = target_spikes[0].cpu().T

clean_logstft_vis = clean_logstft[0].cpu().T  # [F, T]
noisy_logstft_vis = noisy_logstft[0].cpu().T  # [F, T]

log_min_val = log_min[0].item()
log_max_val = log_max[0].item()


denorm_clean_logstft = clean_logstft_vis * (log_max_val - log_min_val) + log_min_val
denorm_target_logstft = ground_truth_vis * (log_max_val - log_min_val) + log_min_val

original_length = original_length[0].item()





waveform = logstft_to_waveform(denorm_clean_logstft, n_fft=1024, hop_length=256, sample_rate=22050)

waveform_fixed = F.interpolate(
    waveform.unsqueeze(0), size=original_length, mode="linear", align_corners=False
).squeeze(0)


torchaudio.save("reconstructed_clean_STFT.wav", waveform_fixed, sample_rate=22050)

waveform = logstft_to_waveform(denorm_target_logstft, n_fft=1024, hop_length=256, sample_rate=22050)  

waveform_fixed = F.interpolate(
    waveform.unsqueeze(0), size=original_length, mode="linear", align_corners=False
).squeeze(0)

torchaudio.save("reconstructed_target_STFT.wav", waveform_fixed, sample_rate=22050) 


# Plot all spike and STFT comparisons
plt.figure(figsize=(14, 12))

# (1,1) Predicted spikes
plt.subplot(3, 2, 1)
plt.imshow(pred_spikes, aspect='auto', origin='lower')
plt.title("Predicted Spikes")
plt.xlabel("Time")
plt.ylabel("STFT Bin")

# (1,2) Target spikes
plt.subplot(3, 2, 2)
plt.imshow(target_spikes_vis, aspect='auto', origin='lower')
plt.title("Target Spikes")
plt.xlabel("Time")
plt.ylabel("STFT Bin")

# (2,1) Reconstructed from predicted spikes
plt.subplot(3, 2, 3)
plt.imshow(predicted_vis, aspect='auto', origin='lower')
plt.title("Reconstructed Log-STFT (Predicted)")
plt.xlabel("Time")
plt.ylabel("STFT Bin")

# (2,2) Reconstructed from target spikes
plt.subplot(3, 2, 4)
plt.imshow(ground_truth_vis, aspect='auto', origin='lower')
plt.title("Reconstructed Log-STFT (Target)")
plt.xlabel("Time")
plt.ylabel("STFT Bin")

# (3,1) Ground truth clean log-STFT
plt.subplot(3, 2, 5)
plt.imshow(clean_logstft_vis, aspect='auto', origin='lower')
plt.title("Original Clean Log-STFT")
plt.xlabel("Time")
plt.ylabel("STFT Bin")


plt.tight_layout()
plt.savefig("spike_logstft_comparison.png")


# Plot spiking activity layer-by-layer
layer_names = list(snn.spk_rec.keys())
n_layers = len(layer_names)
fig, axes = plt.subplots(n_layers, 1, figsize=(12, 2 * n_layers), sharex=True)

if n_layers == 1:
    axes = [axes]

for i, (layer_name, rec) in enumerate(snn.spk_rec.items()):
    spikes = rec.permute(1, 0, 2)[0].cpu().numpy().T  # [N, T]
    axes[i].imshow(spikes, aspect='auto', origin='lower', interpolation='none')
    axes[i].set_ylabel(layer_name, rotation=0, labelpad=40)
    axes[i].yaxis.set_label_position("right")
    axes[i].grid(False)

axes[-1].set_xlabel("Time step")
plt.tight_layout()
plt.savefig("spiking_activity_layers.png")
