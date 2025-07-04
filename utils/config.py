from types import SimpleNamespace

cfg = SimpleNamespace(
    sample_rate=16000,
    n_fft=512,
    hop_length=256, 
    max_len=700, 
    encode_mode="delta",  # "delta", "phased_rate"
    model_type="homebrew-delta",  # "homebrew" or "autoencoder"
    threshold=0.003,#0.003,  # Threshold for delta encoding
    spike_threshold=0.02, #0.2 for phased_rate, 0.01 for delta
    normalize=True,
    padding=True,
    n_epochs=3,
    batch_size=2,
    data_root="E:/VSProjects/datasets/audioVCTK",
    n_freq_bins=257,
    n_iter=32,
    max_samples=300 
)

