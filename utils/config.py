from types import SimpleNamespace

cfg = SimpleNamespace(
    sample_rate=16000,
    n_fft=512,
    hop_length=16, 
    max_len=1000, 
    encode_mode="phased_rate",
    model_type="autoencoder",  # "homebrew" or "autoencoder"
    threshold=0.003,
    spike_threshold=0.2,
    normalize=True,
    padding=True,
    n_epochs=3,
    batch_size=2,
    data_root="E:/VSProjects/datasets/audioVCTK",
    n_freq_bins=257,
    n_iter=32,
    max_samples=500 
)

