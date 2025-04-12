import torch
import torchaudio

def decode_spikes_to_audio(spikes, sample_rate=16000, n_fft=256, hop_length=128, n_mels=40):
    """
    Reconstructs a waveform from a spiking-based mel estimate.
    This is a naive approach. If your final layer is truly binary spikes,
    you might get a very sparse or noisy reconstruction.
    """
    spikes = spikes.squeeze(0)  # [T, F]
    mel_estimate = spikes.float()

    # Quick check
    mel_sum = mel_estimate.sum().item()
    print(f"➡️ Decoded spike sum: {mel_sum:.4f}")

    # Normalize to [0..1] for InverseMelScale
    mel_estimate = mel_estimate / (mel_estimate.max() + 1e-6)
    mel_estimate = mel_estimate.T  # => [n_mels, time]

    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate
    )(mel_estimate)

    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length)
    waveform = griffin_lim(inv_mel)
    return waveform
