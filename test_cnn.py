import torch
from torch.utils.data import DataLoader
from data.dataloader import SpikeSpeechEnhancementDataset
from utils.config import cfg
from models.cnn import build_cnn
import matplotlib.pyplot as plt
from utils.audio_utils import reconstruct_without_stretch
import os

def test_cnn(cfg, show_plot=True, save_audio=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load checkpoint
    ckpt_path = "Trained/2025-07-20_02-17-40_none_e1000_len800_rced/checkpoint_epoch_700.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_folder = os.path.basename(os.path.dirname(ckpt_path))  # 2025-07-19_15-00-56_none_e100_len800_rced

    # === Output directory
    output_dir = os.path.join("outputs", "wavs", ckpt_folder)
    os.makedirs(output_dir, exist_ok=True)

    # === Load model
    model = build_cnn(cfg.n_freq_bins).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # === Dataset
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

    # === Inference loop
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

        if save_audio:
            wav_path = os.path.join(output_dir, f"cnn_output_sample{i}.wav")
            reconstruct_without_stretch(
                logstft_tensor=pred,
                log_min=log_min,
                log_max=log_max,
                filename=wav_path,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                sample_rate=cfg.sample_rate,
                n_iter=cfg.n_iter,
                original_length=original_length
            )
            print(f"Audio saved to {wav_path}")

        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].imshow(y_true.cpu().squeeze().numpy(), origin='lower', aspect='auto')
            axes[0].set_title("Ground Truth")
            axes[1].imshow(pred.numpy(), origin='lower', aspect='auto')
            axes[1].set_title("Predicted Output")
            plt.tight_layout()

            if save_audio:
                fig_path = os.path.join(output_dir, f"cnn_output_sample{i}.png")
                plt.savefig(fig_path, dpi=150)
                print(f"Plot saved to {fig_path}")
            plt.show()

        break  # sadece ilk örnek

test_cnn(cfg)
