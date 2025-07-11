import torch
from torch.utils.data import DataLoader
from data.dataloader import SpikeSpeechEnhancementDataset
from utils.config import cfg
from models.cnn import build_cnn
import matplotlib.pyplot as plt
from utils.audio_utils import reconstruct_without_stretch  # az önce verdiğin fonksiyon
import os

def test_cnn(cfg, show_plot=True, save_audio=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = SpikeSpeechEnhancementDataset(
        noisy_dir=f"{cfg.data_root}/noisy",
        clean_dir=f"{cfg.data_root}/clean",
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        max_len=cfg.max_len,
        threshold=cfg.threshold,
        mode=cfg.encode_mode,
        normalize=cfg.normalize,
        padding=cfg.padding,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = build_cnn(cfg.n_freq_bins).to(device)
    model.load_state_dict(torch.load("cnn_weights_stft.pth"))
    model.eval()

    os.makedirs("outputs", exist_ok=True)

    for i, batch in enumerate(loader):
        _, _, clean_normed, noisy_normed, log_min, log_max, original_length, _ = batch
        x = noisy_normed.to(device).permute(0, 2, 1)  # [1, F, T]
        y_true = clean_normed.to(device).permute(0, 2, 1)

        with torch.no_grad():
            y_pred = model(x)

        pred = y_pred.squeeze(0).cpu()  # [F, T]
        log_min = log_min[0].item()
        log_max = log_max[0].item()
        original_length = original_length[0].item()

        # === Ses kaydı ===
        if save_audio:
            reconstruct_without_stretch(
                logstft_tensor=pred,
                log_min=log_min,
                log_max=log_max,
                filename=f"outputs/cnn_output_sample{i}.wav",
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                sample_rate=cfg.sample_rate,
                n_iter=cfg.n_iter,
                original_length=original_length
            )
            print(f"[✓] Audio saved to outputs/cnn_output_sample{i}.wav")

        # === Görselleştirme ===
        if show_plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(y_true.cpu().squeeze().numpy(), origin='lower', aspect='auto')
            plt.title("Ground Truth")
            plt.subplot(1, 2, 2)
            plt.imshow(pred.numpy(), origin='lower', aspect='auto')
            plt.title("Predicted Output")
            plt.tight_layout()
            plt.show()
        break  # sadece ilk örnek

test_cnn(cfg)
