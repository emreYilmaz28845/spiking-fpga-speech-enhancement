import torch
import torchaudio
import numpy as np

def load_and_encode_spikes(
    audio_path, 
    n_mels=40,
    max_timesteps=73,
    return_mel=False,
    sample_rate=16000,
    n_fft=512,
    hop_length=128,
    spike_threshold_ratio=0.4
):
    """
    Loads an audio file, extracts/normalizes mel, and either returns:
      (spikes, None) when return_mel=False,
      (None, mel) when return_mel=True.
    
    This ensures we always return exactly 2 items.
    """
    # 1) Load audio with validation
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.abs().max() < 1e-6:  # Check for silent audio
            raise ValueError(f"Silent audio file: {audio_path}")
    except Exception as e:
        print(f"Error loading {audio_path}: {str(e)}")
        return None, None

    # 2) Resample if needed
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # 3) Extract Mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    mel = mel_transform(waveform).squeeze(0).T  # shape: [time, n_mels]
    #print(f"🔬 Raw mel time frames before padding: {mel.shape[0]}")
    # Handle empty or near-silent mel
    if mel.numel() == 0 or mel.abs().max() < 1e-6:
        print(f"Empty/invalid mel spectrogram: {audio_path}")
        return None, None

    # 4) Time-axis padding/trimming to max_timesteps
    if mel.shape[0] < max_timesteps:
        pad_amount = max_timesteps - mel.shape[0]
        mel = torch.nn.functional.pad(
            mel, (0, 0, 0, pad_amount), mode='constant', value=1e-6
        )
    else:
        mel = mel[:max_timesteps, :]

    # 5) Return path #1: Clean mel (return_mel=True)
    if return_mel:
        # Clean target: log scaling + normalization + (sigmoid-ish)
        mel = torch.log(mel + 1e-6)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = torch.sigmoid(mel) * 0.9 + 0.05  # scale to [0.05, 0.95]

        # We only care about the mel for the "clean" branch => (None, mel)
        return None, mel

    # 6) Return path #2: Spike encoding for noisy input (return_mel=False)
    #    (a) power-law compression
    mel = torch.pow(mel + 1e-6, 0.3)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    #    (b) adaptive threshold to make binary spikes
    dynamic_threshold = mel.mean() + spike_threshold_ratio * mel.std()
    spikes = (mel > dynamic_threshold).float()

    spikes = torch.nn.functional.avg_pool1d(
        spikes.unsqueeze(0).permute(0, 2, 1),  # [1, F, T]
        kernel_size=3,
        stride=1,           # <-- this preserves time steps
        padding=1
    ).permute(0, 2, 1).squeeze(0)  # => [T=73, F=40]


    # Return shape: [1, T, F] for spiking data
    #print(f"🎛 Final mel shape before spike encoding: {mel.shape}")  # Expect [73, 40]
    #print(f"⚡ Spike shape before unsqueeze: {spikes.shape}")        # Expect [73, 40]
    spikes = spikes.unsqueeze(0)  # Shape should now be [1, 73, 40]
    #print(f"📦 Returning spike tensor with shape: {spikes.shape}")
    return spikes, None

