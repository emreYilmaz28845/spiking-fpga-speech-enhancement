import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.dataloader import SpikeSpeechEnhancementDataset
from utils.config import cfg  
from models.cnn import build_cnn
import matplotlib.pyplot as plt
from utils.loss_stft import STFTLogLoss
import os

os.makedirs("checkpoints/CNN", exist_ok=True)

# --- Validation Loss function ---
def validate(model, val_loader, device, criterion):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            _, _, clean_normed, noisy_normed, *_ = batch
            x_val = noisy_normed.to(device).permute(0, 2, 1)  # [B, F, T]
            y_val = clean_normed.to(device).permute(0, 2, 1)
            out_val = model(x_val)
            loss_val = criterion(out_val, y_val)
            total_val_loss += loss_val.item()
    return total_val_loss / len(val_loader)

def train_cnn(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset creation
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
    
    if hasattr(cfg, "max_samples") and cfg.max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(cfg.max_samples, len(dataset))))
    
    

    # Train / Validation split
    total_samples = len(dataset)
    n_val = int(0.1 * total_samples)
    n_train = total_samples - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"Using {len(dataset)} total samples: {n_train} for training, {n_val} for validation.")
    # Model, optimizer, loss
    model = build_cnn(cfg.n_freq_bins).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = STFTLogLoss()

    train_losses = []
    val_losses = []

    model.train()
    for epoch in range(cfg.n_epochs):
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            _, _, clean_normed, noisy_normed, *_ = batch
            x = noisy_normed.to(device).permute(0, 2, 1)  # [B, F, T]
            y = clean_normed.to(device).permute(0, 2, 1)
    
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            print(f"[Epoch {epoch+1}][Batch {i+1}] Loss: {loss.item():.4f}")
    
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
    
        avg_val_loss = validate(model, val_loader, device, criterion)
        val_losses.append(avg_val_loss)
    
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}")
    
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/CNN/cnn_weights_epoch_{epoch+1}.pth")
    
        # Plot and save loss curve
        plt.figure()
        plt.plot(range(1, epoch+2), train_losses, label="Train Loss")
        plt.plot(range(1, epoch+2), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Progress")
        plt.savefig("loss_progress.png")
        plt.close()
    
    print("Training completed. Loss curve saved as 'loss_progress.png'.")

if __name__ == "__main__":
    train_cnn(cfg)
