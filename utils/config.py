from types import SimpleNamespace

cfg = SimpleNamespace(
    sample_rate=16000,
    n_fft=512,
    hop_length=256,  
    max_len=800, 
    encode_mode="none",  # "delta", "phased_rate"
    model_type="cnn-to-snn-2",  # "homebrew" or "autoencoder" or "dpsnn"
    threshold=0.003,#0.003,  # Threshold for delta encoding
    spike_threshold=0.2, #0.2 for phased_rate, 0.01 for delta
    normalize=True,
    padding=True,
    n_epochs=50,
    batch_size=32,
    data_root="E:/VSProjects/datasets/audioVCTK",
    n_freq_bins=257,
    n_iter=32,
    max_samples=None,
    load_cnn_weights = True,
    use_preencoded= False,
    use_preencoded_noEncode=False,
)

