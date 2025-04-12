import os
import torch
import logging
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from torch.optim import Adam

from preprocess import load_and_encode_spikes
from model import build_snn
from dataset import SpikeSpeechDataset

logging.basicConfig(level=logging.INFO)

def main():
    # === Config ===
    data_dir = "audio"
    batch_size = 16
    epochs = 10
    learning_rate = 1e-3
    thresholds = [0.005, 0.01, 0.02]  # Çoklu eşikler
    n_mels = 40

    # === Load dataset ===
    logging.info("Loading dataset...")
    dataset = SpikeSpeechDataset(data_dir, thresholds=thresholds, n_mels=n_mels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    logging.info(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")

    # === Build model ===
    sample_input, _ = dataset[0]
    T, F = sample_input.shape[0], sample_input.shape[1]  # F already includes all thresholds

    model = build_snn(n_inputs=F, n_outputs=n_mels, n_hidden=128, n_cycles=T)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = MSELoss()

    # === Training loop ===
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x = x.permute(1, 0, 2).to(device)  # [T, B, F]
            y = y.permute(1, 0, 2).to(device)  # [T, B, F]

            optimizer.zero_grad()
            output = model(x)
            out_mem = model.mem_rec["lif6"]  # Output layer
            loss = loss_fn(out_mem, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

    # === Save trained weights ===
    os.makedirs("Trained", exist_ok=True)
    torch.save(model.state_dict(), "Trained/enhancement_model.pt")
    logging.info("Model saved to Trained/enhancement_model.pt")

if __name__ == "__main__":
    main()
