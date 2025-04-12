import os
import torch
from torch.utils.data import Dataset
from preprocess import load_and_encode_spikes

class SpikeSpeechDataset(Dataset):
    def __init__(self, data_root, n_mels=40, max_timesteps=73):
        """
        data_root should contain subfolders: 
          - data_root/noisy/*.wav
          - data_root/clean/*.wav
        Each .wav file in "noisy" must have a matching .wav file in "clean".
        """
        self.noisy_dir = os.path.join(data_root, "noisy")
        self.clean_dir = os.path.join(data_root, "clean")
        self.n_mels = n_mels
        self.max_timesteps = max_timesteps

        self.file_names = sorted([
            f for f in os.listdir(self.noisy_dir)
            if f.endswith(".wav") and os.path.isfile(os.path.join(self.clean_dir, f))
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]
        noisy_path = os.path.join(self.noisy_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        # 1) Load spikes from the noisy audio => (spikes, None)
        noisy_spikes, _ = load_and_encode_spikes(
            noisy_path,
            n_mels=self.n_mels,
            max_timesteps=self.max_timesteps,
            return_mel=False
        )

        # 2) Load mel from the clean audio => (None, mel)
        _, clean_mel = load_and_encode_spikes(
            clean_path,
            n_mels=self.n_mels,
            max_timesteps=self.max_timesteps,
            return_mel=True
        )

        # 3) Check for None returns
        if noisy_spikes is None or clean_mel is None:
            raise ValueError(f"Audio file {filename} gave invalid spikes or mel.")

        # noisy_spikes shape => [1, T, F]
        # clean_mel shape => [T, F]
        return noisy_spikes.squeeze(0), clean_mel
