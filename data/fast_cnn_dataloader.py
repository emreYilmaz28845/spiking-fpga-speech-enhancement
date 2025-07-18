import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class SpeechSpectrogramDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.noisy_dir = os.path.join(cfg.data_root, "noisy")
        self.clean_dir = os.path.join(cfg.data_root, "clean")

        self.sample_rate = cfg.sample_rate
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.max_len = cfg.max_len
        self.padding = cfg.padding
        self.normalize = cfg.normalize

        self.filenames = sorted([
            f for f in os.listdir(self.clean_dir)
            if f.endswith(".wav") and os.path.exists(os.path.join(self.noisy_dir, f))
        ])

        if hasattr(cfg, "max_samples") and cfg.max_samples is not None:
            self.filenames = self.filenames[:cfg.max_samples]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        clean_path = os.path.join(self.clean_dir, fname)
        noisy_path = os.path.join(self.noisy_dir, fname)

        y_clean, _ = librosa.load(clean_path, sr=self.sample_rate)
        y_noisy, _ = librosa.load(noisy_path, sr=self.sample_rate)

        stft_clean = librosa.stft(y_clean, n_fft=self.n_fft, hop_length=self.hop_length)
        stft_noisy = librosa.stft(y_noisy, n_fft=self.n_fft, hop_length=self.hop_length)

        log_clean = np.log1p(np.abs(stft_clean))
        log_noisy = np.log1p(np.abs(stft_noisy))

        log_clean, log_noisy = self._pad_or_truncate(log_clean, log_noisy)

        log_clean = log_clean.T
        log_noisy = log_noisy.T

        if self.normalize:
            log_min = min(log_clean.min(), log_noisy.min())
            log_max = max(log_clean.max(), log_noisy.max())
            log_clean = (log_clean - log_min) / (log_max - log_min + 1e-8)
            log_noisy = (log_noisy - log_min) / (log_max - log_min + 1e-8)
        else:
            log_min, log_max = 0.0, 1.0

        return (
            torch.from_numpy(log_noisy).float(),
            torch.from_numpy(log_clean).float(),
            torch.tensor(log_min).float(),
            torch.tensor(log_max).float(),
        )

    def _pad_or_truncate(self, clean, noisy):
        T = clean.shape[1]
        if T < self.max_len:
            pad_width = self.max_len - T
            clean = np.pad(clean, ((0, 0), (0, pad_width)), mode='constant')
            noisy = np.pad(noisy, ((0, 0), (0, pad_width)), mode='constant')
        elif T > self.max_len:
            clean = clean[:, :self.max_len]
            noisy = noisy[:, :self.max_len]
        return clean, noisy


import os
import torch
from torch.utils.data import Dataset

class PreEncodedSpectrogramDataset(Dataset):
    def __init__(self, preprocessed_dir):
        self.paths = sorted([
            os.path.join(preprocessed_dir, f)
            for f in os.listdir(preprocessed_dir) if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)
        return (
            data["noisy_normed"],   # input (T, F)
            data["clean_normed"],   # target (T, F)
            data["log_min"],
            data["log_max"],
        )
