import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import logging
import matplotlib.pyplot as plt

from spikerplus import NetBuilder, Trainer
from dataloader import SpikeSpeechEnhancementDataset
# === Logging ===
logging.basicConfig(level=logging.INFO)


# === 2. Load Dataset ===
data_root = "C:/VSProjects/spiking-fpga-project/audio"
dataset = SpikeSpeechEnhancementDataset(
    noisy_dir=os.path.join(data_root, "noisy"),
    clean_dir=os.path.join(data_root, "clean"),
    delta_threshold=0.003
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

# === 3. Network ===
example_input, _ = dataset[0]
net_dict = {
    "n_cycles": example_input.shape[0],
    "n_inputs": example_input.shape[1],
    "layer_0": {"neuron_model": "lif", "n_neurons": 128, "beta": 0.9375, "threshold": 0.01, "reset_mechanism": "subtract"},
    "layer_1": {"neuron_model": "lif", "n_neurons": 128, "beta": 0.9375, "threshold": 0.01, "reset_mechanism": "subtract"},
    "layer_2": {"neuron_model": "lif", "n_neurons": 40,  "beta": 0.9375, "threshold": 0.01, "reset_mechanism": "none"}
}
snn = NetBuilder(net_dict).build()

# === 4. Train ===
trainer = Trainer(snn)
trainer.train(train_loader, val_loader, n_epochs=1, store=True, output_dir="trained_models")

# === 5. Loss Plot
batch_losses = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in trainer.batch_losses]
plt.figure(figsize=(10, 4))
plt.plot(batch_losses, label="Batch Loss")
plt.xlabel("Batch")
plt.ylabel("Loss (MSE)")
plt.title("Batch-wise Training Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === 6. Inference Visualization ===
snn.eval()
sample_batch = next(iter(val_loader))
input_spikes, target = sample_batch
input_spikes = input_spikes.permute(1, 0, 2).to(trainer.device)  # [T, B, F]
target = target.to(trainer.device)

with torch.no_grad():
    snn(input_spikes)
    _, mem_out = list(snn.mem_rec.items())[-1]
    mem_out = mem_out.permute(1, 0, 2)  # [B, T, F]


print("Output range:", mem_out.min().item(), mem_out.max().item())
print("Target range:", target.min().item(), target.max().item())

# Normalize membrane output if needed
mem_out = (mem_out - mem_out.mean()) / (mem_out.std() + 1e-6)


# Plot one sample
predicted = mem_out[0].cpu().T
ground_truth = target[0].cpu().T

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(predicted, aspect='auto', origin='lower')
plt.title("Predicted Membrane Output")
plt.subplot(1, 2, 2)
plt.imshow(ground_truth, aspect='auto', origin='lower')
plt.title("Target Clean Log-Mel")
plt.tight_layout()
plt.show()

