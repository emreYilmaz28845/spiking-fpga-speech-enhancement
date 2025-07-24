from types import SimpleNamespace

cfg = SimpleNamespace(
    sample_rate=16000,
    n_fft=512,
    hop_length=32,  
    max_len=4000, 
    encode_mode="phased_rate",  # "delta", "phased_rate"
    model_type="spiking-fsb-conv-filter-predict",  # "homebrew" or "autoencoder" or "dpsnn"
    threshold=0.003,#0.003,  # Threshold for delta encoding
    spike_threshold=0.2, #0.2 for phased_rate, 0.01 for delta
    normalize=True,
    padding=True,
    n_epochs=20,
    batch_size=32,
    data_root="E:/VSProjects/datasets/audioVCTK",
    n_freq_bins=257,
    n_iter=32,
    max_samples=1000,
    load_cnn_weights = False, 
    use_preencoded= False,
    use_preencoded_noEncode=False,
    predict_filter = True
)

