import os
import time
import logging
import torch
import torch.nn as nn


#change the trainer in the env
logging.basicConfig(level=logging.DEBUG)

class Trainer:
    def __init__(self, net, optimizer=None, loss_fn=None):
        self.net = net

        self.loss_fn = loss_fn if loss_fn else nn.MSELoss()

        if not optimizer:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-4, betas=(0.9, 0.999))
        else:
            self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        logging.info("Initialized EnhancementTrainer")
        logging.info(f"Loss function: {self.loss_fn}")
        logging.info(f"Optimizer: {self.optimizer}")
        logging.info(f"Device: {self.device}")

    def train(self, train_loader, val_loader, n_epochs=20, store=False, output_dir="Trained"):
        logging.info(f"Epochs: {n_epochs}")
        logging.info(f"Training samples: {len(train_loader.dataset)}")
        logging.info(f"Validation samples: {len(val_loader.dataset)}")
        logging.info("Begin training\n")

        start_time = time.time()

        for epoch in range(n_epochs):
            logging.info(f"Epoch {epoch+1}/{n_epochs} started")
            train_loss = self.train_one_epoch(train_loader)
            logging.info(f"Epoch {epoch+1} training done. Starting evaluation...")
            val_loss = self.evaluate(val_loader)
            self.log(epoch, train_loss, val_loss, start_time)

        if store:
            self.store(output_dir)

    def train_one_epoch(self, dataloader):
        total_loss = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            logging.debug(f"Batch {batch_idx}: Input shape {data.shape}, Target shape {targets.shape}")
            data = data.permute(1, 0, 2).to(self.device)  # [T, B, F]
            targets = targets.to(self.device)  # [B, T, F]

            self.optimizer.zero_grad()
            self.net.train()
            self.net(data)

            _, mem_out = list(self.net.mem_rec.items())[-1]  # [T, B, F]
            mem_out = mem_out.permute(1, 0, 2)  # [B, T, F]

            loss_val = self.loss_fn(mem_out, targets)
            loss_val.backward()
            self.optimizer.step()

            logging.debug(f"Batch {batch_idx}: Loss = {loss_val.item():.4f}")
            total_loss += loss_val.item()

        avg_loss = total_loss / (batch_idx + 1)
        logging.info(f"Training epoch done. Avg loss = {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, dataloader):
        total_loss = 0
        with torch.no_grad():
            self.net.eval()

            for batch_idx, (data, targets) in enumerate(dataloader):
                data = data.permute(1, 0, 2).to(self.device)  # [T, B, F]
                targets = targets.to(self.device)  # [B, T, F]

                self.net(data)
                _, mem_out = list(self.net.mem_rec.items())[-1]  # [T, B, F]
                mem_out = mem_out.permute(1, 0, 2)  # [B, T, F]

                loss_val = self.loss_fn(mem_out, targets)
                logging.debug(f"Validation Batch {batch_idx}: Loss = {loss_val.item():.4f}")
                total_loss += loss_val.item()

        avg_loss = total_loss / (batch_idx + 1)
        logging.info(f"Validation done. Avg loss = {avg_loss:.4f}")
        return avg_loss

    def log(self, epoch, train_loss, val_loss, start_time=None):
        elapsed = time.time() - start_time if start_time else 0.0
        log_message = f"Epoch {epoch+1}\n"
        log_message += f"Elapsed time: {elapsed:.2f}s\n"
        log_message += f"Train loss: {train_loss:.4f}\n"
        log_message += f"Validation loss: {val_loss:.4f}\n"
        logging.info(log_message)

    def store(self, out_dir, out_file="trained_state_dict.pt"):
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_file)
        torch.save(self.net.state_dict(), out_path)
        logging.info(f"Model saved to {out_path}")
