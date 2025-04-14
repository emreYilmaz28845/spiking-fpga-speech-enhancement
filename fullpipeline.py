import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from spikerplus import NetBuilder
from spikerplus import Trainer  # Use the rewritten trainer class

# === 1. Custom Dataset for Speech Enhancement ===
class SpikeSpeechEnhancementDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, sample_rate=16000, n_fft=512, hop_length=128, n_mels=40, max_len=1500):
        self.noisy_paths = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        self.clean_paths = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.n_mels = n_mels
        self.max_len = max_len

    def pad_or_truncate(self, x):
        T = x.shape[0]
        if T > self.max_len:
            return x[:self.max_len]
        else:
            pad = torch.zeros(self.max_len - T, x.shape[1])
            return torch.cat([x, pad], dim=0)

    def __len__(self):
        return len(self.noisy_paths)

    def rate_code(self, mel_tensor, T=1500):
        """
        Rate code the mel tensor into binary spike trains.
        Args:
            mel_tensor: [n_mels, T_orig]
        Returns:
            spikes: [T, n_mels] binary tensor
        """
        mel = torch.pow(mel_tensor + 1e-6, 0.3)
        mel = (mel - mel.min()) / (mel.max() - mel.min())

        # Interpolate to fixed length T
        mel = torch.nn.functional.interpolate(mel.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0)  # [n_mels, T]
        mel = mel.T  # [T, n_mels]

        # Rate coding: spike if rand < value
        spikes = torch.bernoulli(mel)
        return spikes

    def __getitem__(self, idx):
        noisy_wave, _ = torchaudio.load(self.noisy_paths[idx])
        clean_wave, _ = torchaudio.load(self.clean_paths[idx])

        noisy_mel = self.mel_transform(noisy_wave).squeeze(0)  # [n_mels, T]
        clean_mel = self.mel_transform(clean_wave).squeeze(0)  # [n_mels, T]

        # Apply power-law compression and normalization
        clean_mel = torch.pow(clean_mel + 1e-6, 0.3)
        clean_mel = (clean_mel - clean_mel.mean()) / (clean_mel.std() + 1e-6)

        spikes = self.rate_code(noisy_mel, T=self.max_len)              # [T, n_mels]
        clean = self.pad_or_truncate(clean_mel.T)                       # [T, n_mels]

        return spikes, clean



# === 2. Load Dataset ===
data_root = "C:/VSProjects/spiking-fpga-project/audio"
noisy_dir = os.path.join(data_root, "noisy")
clean_dir = os.path.join(data_root, "clean")
dataset = SpikeSpeechEnhancementDataset(noisy_dir, clean_dir)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

# === 3. Define SNN Model ===
example_input, _ = dataset[0]
net_dict = {
    "n_cycles": example_input.shape[0],
    "n_inputs": example_input.shape[1],
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

# === 4. Train ===
trainer = Trainer(snn)
trainer.train(train_loader, val_loader, n_epochs=20, store=True, output_dir="trained_models")
