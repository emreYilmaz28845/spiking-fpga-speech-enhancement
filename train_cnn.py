import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from data.dataloader import PreEncodedDataset
from data.fast_cnn_dataloader import SpeechSpectrogramDataset,PreEncodedSpectrogramDataset
from models.cnn import build_cnn
from utils.loss_stft import STFTLogLoss
from utils.config import cfg
from data.dataloader import SpikeSpeechEnhancementDataset

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
            else:
                noisy_normed, clean_normed, *_ = batch

            x_val = noisy_normed.to(device).permute(0, 2, 1)
            y_val = clean_normed.to(device).permute(0, 2, 1)
            out_val = model(x_val)
            loss_val = criterion(out_val, y_val)
            total_val_loss += loss_val.item()
    return total_val_loss / len(val_loader)

def train_cnn(cfg, resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")


    if cfg.use_preencoded:
        print("🔁 Using pre-encoded spike dataset...")
        base_dir = f"{cfg.data_root}/preprocessed"
        dataset_dir = make_out_dir(cfg, base_dir)
        dataset = PreEncodedDataset(dataset_dir)
    elif cfg.use_preencoded_noEncode:
        print("🧠 Using pre-encoded log-STFT spectrogram dataset...")
        base_dir = f"{cfg.data_root}/preprocessed"
        dataset_dir = make_out_dir(cfg, base_dir)
        dataset = PreEncodedSpectrogramDataset(dataset_dir)
    else:
        print("🎙 Using raw waveform dataset...")
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


    # === max_samples kısıtlaması ===
    if hasattr(cfg, "max_samples") and cfg.max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(cfg.max_samples, len(dataset))))

    total_samples = len(dataset)
    n_val = int(0.1 * total_samples)
    n_train = total_samples - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Using {len(dataset)} total samples: {n_train} for training, {n_val} for validation.")

    model = build_cnn(cfg.n_freq_bins).to(device)
    print(f"VRAM usage after model to(device): {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    criterion = STFTLogLoss().to(device)

    # === 🔍 VRAM testi için ilk batch ===
    example_batch = next(iter(train_loader))
    if cfg.use_preencoded:
        _, _, clean_normed, noisy_normed, *_ = example_batch
    else:
        noisy_normed, clean_normed, *_ = example_batch

    x = noisy_normed.to(device).permute(0, 2, 1)
    y = clean_normed.to(device).permute(0, 2, 1)

    print(f"VRAM after first batch: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    train_losses = []
    val_losses = []
    lr_list = []
    start_epoch = 0

    # === Checkpoint'ten devam et ===
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    # === Eğitim Döngüsü ===
    model.train()
    for epoch in range(start_epoch, cfg.n_epochs):
        start_time = time.time()
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            if cfg.use_preencoded:
                _, _, clean_normed, noisy_normed, *_ = batch
            else:
                noisy_normed, clean_normed, *_, mask= batch

            x = noisy_normed.to(device).permute(0, 2, 1)
            y = clean_normed.to(device).permute(0, 2, 1)

            if mask is not None:
                mask = mask.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            if mask is not None:
                loss = criterion(out, y, mask=mask)
            else: 
                loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"[Epoch {epoch+1}][Batch {i+1}] Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = validate(model, val_loader, device, criterion)
        val_losses.append(avg_val_loss)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_list.append(new_lr)  
        if new_lr < old_lr:
            print(f"⚠️ Learning rate decreased: {old_lr:.6f} → {new_lr:.6f}")
        else:
            print(f"📉 Current Learning Rate: {new_lr:.6f}")

        elapsed = time.time() - start_time
        eta = (cfg.n_epochs - epoch - 1) * elapsed
        eta_minutes, eta_seconds = int(eta // 60), int(eta % 60)
        

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}")
        print(f"⏳ ETA: ~{eta_minutes}m {eta_seconds}s remaining")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"📉 Current Learning Rate: {current_lr:.6f}")

        # === Checkpoint kaydet ===
        os.makedirs("checkpoints/CNN", exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, f"checkpoints/CNN/checkpoint_epoch_{epoch+1}.pth")

        # === Kayıp grafiği çiz ===
        plt.figure()
        plt.plot(range(1, epoch+2), train_losses, label="Train Loss")
        plt.plot(range(1, epoch+2), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Progress")
        plt.savefig("loss_progress.png")
        plt.close()

        # === LR grafiği çiz ===
        plt.figure()
        plt.plot(range(1, len(lr_list) + 1), lr_list, label="Learning Rate", color='green')
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.savefig("lr_schedule.png")
        plt.close()


    print("✅ Training completed. Loss curve saved as 'loss_progress.png'.")

if __name__ == "__main__":
    train_cnn(cfg)
