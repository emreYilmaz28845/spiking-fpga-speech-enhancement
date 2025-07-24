import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime
import json
from data.dataloader import PreEncodedDataset, SpikeSpeechEnhancementDataset
from data.fast_cnn_dataloader import SpeechSpectrogramDataset, PreEncodedSpectrogramDataset
from models.cnn import build_cnn
from utils.loss_stft import STFTLogWaveformLoss, STFTMagLoss  # Waveform destekli loss
from utils.config import cfg



def get_run_name(cfg):
    time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parts = [
        cfg.encode_mode,
        f"e{cfg.n_epochs}",
        f"len{cfg.max_len}",
        f"{cfg.model_type}",
    ]
    return time_tag + "_" + "_".join(parts)


def make_out_dir(cfg, base_dir):
    out_name = f"{cfg.encode_mode}_Hop={cfg.hop_length}_Length={cfg.max_len}_NFFT={cfg.n_fft}"
    if getattr(cfg, "use_preencoded_noEncode", False):
        out_name += "_NoEncoding"
    return os.path.join(base_dir, out_name)

def validate(model, val_loader, device, criterion):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            if cfg.use_preencoded:
                _, _, clean_normed, noisy_normed, *_ = batch
                x_val = noisy_normed.to(device).permute(0, 2, 1)
                y_val = clean_normed.to(device).permute(0, 2, 1)

                out_val = model(x_val)  # 🔧 Model çalıştırılıyor
                if out_val.size(2) > y_val.size(2):
                    out_val = out_val[:, :, :y_val.size(2)]
                elif out_val.size(2) < y_val.size(2):
                    y_val = y_val[:, :, :out_val.size(2)]

                loss_val = criterion(out_val, y_val)

            else:
                _, _, clean_normed, noisy_normed, log_min, log_max, original_length, mask = batch
                x_val = noisy_normed.to(device).permute(0, 2, 1)
                y_val = clean_normed.to(device).permute(0, 2, 1)
                mask = mask.to(device)
                log_min = log_min.to(device)
                log_max = log_max.to(device)
                original_length = original_length.to(device)

                out_val = model(x_val)  # 🔧 Model yine çalıştırılıyor
                if out_val.size(2) > y_val.size(2):
                    out_val = out_val[:, :, :y_val.size(2)]
                elif out_val.size(2) < y_val.size(2):
                    y_val = y_val[:, :, :out_val.size(2)]

                loss_val = criterion(out_val, y_val, mask, log_min, log_max, original_length)

            total_val_loss += loss_val.item()
    return total_val_loss / len(val_loader)


def train_cnn(cfg, resume_path=None):
    import matplotlib.pyplot as plt
    from datetime import datetime

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # === Run name & output directory ===
    run_name = get_run_name(cfg)
    run_dir = os.path.join("Trained", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # === Dataset ===
    if cfg.use_preencoded:
        dataset = PreEncodedDataset(make_out_dir(cfg, f"{cfg.data_root}/preprocessed"))
    elif cfg.use_preencoded_noEncode:
        dataset = PreEncodedSpectrogramDataset(make_out_dir(cfg, f"{cfg.data_root}/preprocessed"))
    else:
        dataset = SpikeSpeechEnhancementDataset(
            noisy_dir=f"{cfg.data_root}/noisy",
            clean_dir=f"{cfg.data_root}/clean",
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            max_len=cfg.max_len,
            threshold=cfg.threshold,
            normalize=cfg.normalize,
            mode=cfg.encode_mode,
            padding=cfg.padding,
        )

    if cfg.max_samples:
        dataset = torch.utils.data.Subset(dataset, range(min(cfg.max_samples, len(dataset))))

    n_val = int(0.1 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"Using {len(dataset)} samples → Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # === Model, Loss, Optimizer ===
    model = build_cnn(cfg.n_freq_bins).to(device)
    print(f"Param count: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
    print(f"VRAM after to(device): {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    criterion = STFTLogWaveformLoss(cfg=cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, threshold=1e-4)

    train_losses, val_losses, lr_list = [], [], []
    start_epoch = 0

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, cfg.n_epochs):
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            if cfg.use_preencoded:
                _, _, clean, noisy, *_ = batch
                x = noisy.to(device).permute(0, 2, 1)
                y = clean.to(device).permute(0, 2, 1)
                out = model(x)
                loss = criterion(out[..., :y.size(2)], y[..., :out.size(2)])
            else:
                _, _, clean, noisy, log_min, log_max, original_len, mask = batch
                x = noisy.to(device).permute(0, 2, 1)
                y = clean.to(device).permute(0, 2, 1)
                loss = criterion(model(x), y, mask.to(device), log_min.to(device), log_max.to(device), original_len.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"[Epoch {epoch+1}][Batch {i+1}] Loss: {loss.item():.4f}")

        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)

        avg_val = validate(model, val_loader, device, criterion)
        val_losses.append(avg_val)

        scheduler.step(avg_val)
        lr_list.append(optimizer.param_groups[0]['lr'])

        print(f"[Epoch {epoch+1}] Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {lr_list[-1]:.2e}")

        # === Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth"))

        # === Save figures
        def save_plot(data, title, ylabel, path):
            plt.figure()
            plt.plot(range(1, len(data)+1), data)
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.savefig(path)
            plt.close()

        save_plot(train_losses, "Training Loss", "Loss", os.path.join(run_dir, "train_loss.png"))
        save_plot(val_losses, "Validation Loss", "Loss", os.path.join(run_dir, "val_loss.png"))
        save_plot(lr_list, "Learning Rate", "LR", os.path.join(run_dir, "lr_schedule.png"))

        # === Save logs
        with open(os.path.join(run_dir, "log.txt"), "w") as f:
            for i, (t, v, l) in enumerate(zip(train_losses, val_losses, lr_list)):
                f.write(f"Epoch {i+1}: Train={t:.6f}, Val={v:.6f}, LR={l:.6e}\n")

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(cfg), f, indent=2)

    print(f"\n✅ Training completed. Logs saved to: {run_dir}")


    print("✅ Training completed. Loss curve saved as 'loss_progress.png'.")

if __name__ == "__main__":
    train_cnn(cfg)#,resume_path="Trained/2025-07-20_02-17-40_none_e1000_len800_rced/checkpoint_epoch_780.pth")
