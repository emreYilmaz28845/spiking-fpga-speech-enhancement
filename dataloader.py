from encode import spike_encode  
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torch
import os

# === 1. Dataset ===
class SpikeSpeechEnhancementDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, sample_rate=16000, n_fft=512, hop_length=128,
                 n_mels=40, max_len=1500, delta_threshold=0.003):
        self.noisy_paths = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        self.clean_paths = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)
        self.max_len = max_len
        self.threshold = delta_threshold

    def pad_or_truncate(self, x):
        if x.shape[0] > self.max_len:
            return x[:self.max_len]
        pad = torch.zeros(self.max_len - x.shape[0], x.shape[1])
        return torch.cat([x, pad], dim=0)

    def __getitem__(self, idx):
        noisy_wave, _ = torchaudio.load(self.noisy_paths[idx])
        clean_wave, _ = torchaudio.load(self.clean_paths[idx])

        noisy_mel = self.mel_transform(noisy_wave).squeeze(0)
        clean_mel = self.mel_transform(clean_wave).squeeze(0)

        # 🧠 Spike encode using centralized encode.py
        spikes, _ = spike_encode(noisy_mel, max_len=self.max_len, threshold=self.threshold)

        # 🧠 Target: use same normalization as spike encode, from clean signal
        _, clean_logmel = spike_encode(clean_mel, max_len=self.max_len, threshold=self.threshold)

        return spikes, clean_logmel  # both are [T, F]

    def __len__(self):
        return len(self.noisy_paths)