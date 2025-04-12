import os
import torch
import logging
from torch.utils.data import DataLoader, random_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from model import build_snn
from dataset import SpikeSpeechDataset

logging.basicConfig(level=logging.INFO)

def main():
    # === Config ===
    data_dir = "audio"
    batch_size = 16
    epochs = 10
    learning_rate = 1e-3
    n_mels = 40

    # === Load dataset ===
    logging.info("Loading dataset...")
    dataset = SpikeSpeechDataset(data_dir, n_mels=n_mels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    logging.info(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")

    # === Build model ===
    # Let's see a sample input shape
    sample_input, sample_target = dataset[0]  # This should now work
    T, F = sample_input.shape  # T time-steps, F frequency bins
    print(T)
    # sample_target is also shape [T, F]

    model = build_snn(n_inputs=F, n_outputs=n_mels, n_cycles=T)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = BCEWithLogitsLoss() 
    # ^ if you are treating the output as a "spike mask" or probability. 
    # Consider MSELoss() or L1Loss() if you want a direct regression.

    # === Training loop ===
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            # x shape: [batch, T, F], y shape: [batch, T, F]
            # spikerplus often expects [T, batch, F] => so transpose dims:
            x = x.permute(1, 0, 2).to(device)  # [T, B, F]
            y = y.permute(1, 0, 2).to(device)  # [T, B, F]
            #print(f"⏱️ Input x shape: {x.shape}")  # Should be [73, B, 40]
            #print(f"🎯 Target y shape: {y.shape}")  # Should be [73, B, 40]

            optimizer.zero_grad()
            _ = model(x)

            #print(f"📦 Available spike layers: {list(model.spk_rec.keys())}")

            # spk_rec is a dict of spikes from each layer
            out_spikes = model.spk_rec["lif6"]  # or "lif6" if your net calls it that
            # Make sure you check the actual layer key. 
            # If your final layer is "layer_5", spikerplus might call it "lif6".
            #print(f"🔍 Output out_spikes shape: {out_spikes.shape}") 
            # BCEWithLogitsLoss expects raw logits in [-∞, ∞], 
            # but spiking outputs are often in {0,1}. 
            # If spikerplus does store the membrane potentials in spk_rec, 
            # you might need to clarify. 
            # We'll assume out_spikes is a continuous or partial "logit" value.

            # shape of out_spikes => [T, B, F]
            loss = loss_fn(out_spikes, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                logging.info(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        logging.info(f"[Epoch {epoch+1}] Average Train Loss: {avg_loss:.6f}")

    # === Save model ===
    os.makedirs("Trained", exist_ok=True)
    torch.save(model.state_dict(), "Trained/enhancement_model.pt")
    logging.info("Model saved to Trained/enhancement_model.pt")

if __name__ == "__main__":
    main()
