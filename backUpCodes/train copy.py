import os
import time
import logging
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import SpikePositionLoss
from utils.encode import reconstruct_from_spikes

logging.basicConfig(level=logging.DEBUG)

# ========== Gradient Debugging Hooks ==========
def spike_hook(grad):
    print("⪮ ⪢ surrogate dL/dSpikes mean =", grad.abs().mean().item())
    return grad

def _print_hook(grad):
    print(f"⪮⪢ ∂L/∂S  mean = {grad.mean().item():.6f}, max = {grad.max().item():.6f}")
    return grad

# ========== Trainer Class ==========
class Trainer:
    def __init__(self, net, optimizer=None, loss_params=None):
        self.net = net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        self.loss_fn = SpikePositionLoss(
            tau=5.0,
            lambda_pos=1.0,
            lambda_vr=0.01,
            r_target=None,
            device=self.device
        )

        self.loss_params = loss_params or {}
        self.batch_losses = []
        self.logs = {
            'target_rate':    [],
            'pred_rate':      [],
            'spike_loss':     [],
            'rate_penalty':   [],
            'weight_change':  [],
            'layer_deltas':   {}
        }

        logging.info(f"Initialized Trainer on {self.device}")

    def train(self, train_loader, val_loader, n_epochs=20, store=False, output_dir="Trained", encode_mode=None, max_len=None):
        start_time = time.time()
        logging.info(f"Training for {n_epochs} epochs")

        for epoch in range(1, n_epochs + 1):
            epoch_start = time.time()
            logging.info(f"Epoch {epoch}/{n_epochs} started")

            self.batch_losses.clear()
            train_loss = self.train_one_epoch(train_loader, epoch, n_epochs)
            val_loss = self.evaluate(val_loader)

            elapsed = time.time() - start_time
            avg_epoch_time = elapsed / epoch
            eta_epochs = avg_epoch_time * (n_epochs - epoch)

            self.log_epoch(epoch, train_loss, val_loss, elapsed, eta_epochs)
            self.scheduler.step()

        if store:
            timestamp = time.strftime("%Y-%m-%d_%H-%M")
            folder_name = f"{timestamp}_{encode_mode}_e{n_epochs}_len{max_len}"
            output_dir = os.path.join(output_dir, folder_name)
            os.makedirs(output_dir, exist_ok=True)
            self.save_model(output_dir)
            self.save_logs(out_dir=output_dir)

    def train_one_epoch(self, dataloader, epoch=1, n_epochs=1):
        self.net.train()
        total_loss = 0.0
        start_time = time.time()
        num_batches = len(dataloader)

        for batch_idx, (noisy_spikes, target_spikes, *_, joint_mask) in enumerate(dataloader):
            batch_start = time.time()

            noisy = noisy_spikes.permute(1, 0, 2).to(self.device)
            target = target_spikes.to(self.device)
            mask = joint_mask.to(self.device)

            r_target = (target_spikes > 0).float().mean().item()
            self.logs['target_rate'].append(r_target)
            print("Target firing-rate:", r_target)

            self.optimizer.zero_grad()
            self.net(noisy)

            _, rec = list(self.net.spk_rec.items())[-1]
            spike_out = rec.permute(1, 0, 2)

            rec.register_hook(_print_hook)
            spike_out.register_hook(spike_hook)

            spike_loss = self.loss_fn(spike_out, target, mask=mask)
            mse_loss = F.mse_loss(spike_out, target.float())
            print("Raw MSE:", mse_loss.item())
            print("========================")

            self.logs['spike_loss'].append(spike_loss.item())
            loss = spike_loss
            loss.backward()

            self._log_gradients()
            self._update_weights()

            spike_rate = (spike_out > 0).float().mean().item()
            self.logs['pred_rate'].append(spike_rate)

            logging.debug(
                f"[Epoch {epoch}/{n_epochs} | Batch {batch_idx + 1}/{num_batches}] SpikeRate: {spike_rate:.3f} | SpkLoss: {spike_loss.item():.3f}"
            )

            self.batch_losses.append(loss.item())
            total_loss += loss.item()

            batch_elapsed = time.time() - start_time
            avg_batch_time = batch_elapsed / (batch_idx + 1)
            eta_batches = avg_batch_time * (num_batches - (batch_idx + 1))

            print(f"[Epoch {epoch}/{n_epochs} | Batch {batch_idx + 1}/{num_batches}] ETA: {eta_batches:.1f}s")

        return total_loss / len(dataloader)

    def _log_gradients(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm().item()}")

    def _update_weights(self):
        weight_before = {name: param.data.clone() for name, param in self.net.named_parameters() if param.requires_grad}
        self.optimizer.step()

        for name, param in self.net.named_parameters():
            if param.requires_grad:
                delta = (param.data - weight_before[name]).norm().item()
                self.logs['layer_deltas'].setdefault(name, []).append(delta)
                print(f"{name} change after step: {delta:.6f}")

    def evaluate(self, dataloader):
        self.net.eval()
        total_loss = 0.0

        with torch.no_grad():
            for noisy_spikes, target_spikes, *_, joint_mask in dataloader:
                noisy = noisy_spikes.permute(1, 0, 2).to(self.device)
                target = target_spikes.to(self.device)

                self.net(noisy)
                _, rec = list(self.net.spk_rec.items())[-1]
                spike_out = rec.permute(1, 0, 2)

                loss = self.loss_fn(spike_out, target)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def log_epoch(self, epoch, train_loss, val_loss, elapsed, eta_epochs):
        logging.info(
            f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Elapsed: {elapsed:.1f}s, ETA: {eta_epochs:.1f}s"
        )

    def save_model(self, out_dir, out_file="trained_state_dict.pt"):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, out_file)
        torch.save(self.net.state_dict(), path)
        logging.info(f"Model state dict saved to {path}")

    def save_logs(self, out_file="logs.json", out_dir="checkpoints"):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, out_file)
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=2)
        logging.info(f"Training logs saved to {path}")

