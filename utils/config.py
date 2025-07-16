from types import SimpleNamespace

cfg = SimpleNamespace(
    sample_rate=16000,
    n_fft=512,
    hop_length=32,  
    max_len=5000, 
    encode_mode="delta",  # "delta", "phased_rate"
    model_type="cnn-to-snn-2",  # "homebrew" or "autoencoder" or "dpsnn"
    threshold=0.003,#0.003,  # Threshold for delta encoding
    spike_threshold=0.2, #0.2 for phased_rate, 0.01 for delta
    normalize=True,
    padding=True,
    n_epochs=50,
    batch_size=2,
    data_root="E:/VSProjects/datasets/audioVCTK",
    n_freq_bins=257,
    n_iter=32,
    max_samples=6000,
    load_cnn_weights = True
)

