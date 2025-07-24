#use it for chaning spiker env trainer.py class
# trainer.py
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import SpikePositionLoss,FilterLoss
from utils.loss_stft import STFTMagLoss
from utils.loss_for_delta import DeltaReconstructionLoss
from utils.encode import reconstruct_from_spikes
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.DEBUG)

def spike_hook(grad):
    print("⪡ ⪢ surrogate dL/dSpikes mean =", grad.abs().mean().item())
    return grad

def _print_hook(grad):
    print(f"⪡⪢ ∂L/∂S  mean = {grad.mean().item():.6f}, max = {grad.max().item():.6f}")
    return grad

class Trainer:
    def __init__(self, net, optimizer=None, loss_params=None, cfg=None, encode_mode=None, predict_filter=False):
        self.net = net
        self.cfg = cfg
        if cfg is not None:
            self.net.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.encode_mode = encode_mode or getattr(cfg, 'encode_mode', 'phased_rate')
        self.predict_filter = predict_filter or getattr(cfg, 'predict_filter', False)

        self.optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4) #1e-4 de kullanılabilir
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.7)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        if self.encode_mode == 'delta':
            print("Using DeltaReconstructionLoss for delta mode")
            self.loss_fn = DeltaReconstructionLoss(
                alpha=0.5,
                gamma_stft=0.5,
                gamma_sisdr=0.001,
                reduction="mean"
            )
        elif self.encode_mode in ['rate', 'phased_rate']:
            print("Using SpikePositionLoss for rate-based modes")
            self.loss_fn = SpikePositionLoss(
                tau=5.0,
                lambda_pos=0.02,
                lambda_vr=0.1,
                gamma_stft=9,  # Temporal focus factor
                reduction="mean",
                r_target=None,
                device=self.device
            )
        if predict_filter:
            print("Using FilterLoss for prediction filter training")
            self.loss_fn = FilterLoss(
                reduction="mean"
            )
        self.loss_params = loss_params or {}
        self.batch_losses = []
        self.logs = {
            'target_rate':    [],
            'pred_rate':      [],
            'spike_loss':     [],
            'rate_penalty':   [],
            'learning_rate': [],
            'epoch':          [],
            'weight_change':  [],
            'gradient_norms': [],
            'layer_deltas':   {}
        }
        logging.info(f"Initialized Trainer on {self.device}")

    def train(self, train_loader, val_loader, n_epochs=5, store=False, output_dir="Trained",
               encode_mode=None, max_len=None, start_epoch=1):
        start_time = time.time()
        logging.info(f"Training for {n_epochs} epochs")

        for epoch in range(start_epoch, n_epochs + 1):
            logging.info(f"Epoch {epoch}/{n_epochs} started")
            epoch_start_time = time.time()

            self.batch_losses.clear()
            print("LR after resume:", self.optimizer.param_groups[0]['lr'])
            train_loss = self.train_one_epoch(train_loader,encode_mode=encode_mode)
            val_loss = self.evaluate(val_loader)
            self.log_epoch(epoch, train_loss, val_loss, start_time)
            self.logs['epoch'].append(epoch)
            self.logs['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.scheduler.step(val_loss)


            self.save_checkpoint(epoch=epoch)

            eta = (time.time() - epoch_start_time) * (n_epochs - epoch)
            logging.info(f"ETA: {eta:.1f} seconds ≈ {eta/60:.1f} minutes remaining")

        if store:
            timestamp = time.strftime("%Y-%m-%d_%H-%M")
            folder_name = f"{timestamp}_{encode_mode}_e{n_epochs}_len{max_len}_arch_{self.net.cfg.model_type}"
            output_dir = os.path.join(output_dir, folder_name)
            os.makedirs(output_dir, exist_ok=True)
            self.save_model(output_dir)
            self.save_logs(out_dir=output_dir)

    def train_one_epoch(self, dataloader,encode_mode=None):
        self.net.train()
        total_loss = 0.0
        total_batches = len(dataloader)
        start_time = time.time()
        if encode_mode == 'delta':
            for batch_idx, (noisy_spikes,target_spikes, *_, log_min, log_max, _, joint_mask) in enumerate(dataloader):
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

                spike_loss = self.loss_fn(spike_out, target, log_min=log_min,log_max=log_max,mask=mask)


                mse_loss = F.mse_loss(spike_out, target.float())
                print("Raw MSE:", mse_loss.item())
                print("========================")

                if hasattr(self.loss_fn, "last_coverage"):
                    self.logs.setdefault("coverage", []).append(self.loss_fn.last_coverage)
                    print(f"Spike coverage: {self.loss_fn.last_coverage:.4f}")

                self.logs['spike_loss'].append(spike_loss.item())
                loss = spike_loss
                loss.backward()

                self._log_gradients()
                self._update_weights()

                spike_rate = (spike_out > 0).float().mean().item()
                self.logs['pred_rate'].append(spike_rate)

                logging.debug(
                    f"[Batch {batch_idx+1}/{total_batches}] SpikeRate: {spike_rate:.3f} | SpkLoss: {spike_loss.item():.3f}"
                )

                self.batch_losses.append(loss.item())
                total_loss += loss.item()

                eta = (time.time() - start_time) / (batch_idx + 1) * (total_batches - (batch_idx + 1))
                print(f"ETA for epoch: {eta:.1f}s ≈ {eta/60:.1f}min")

            
        else:
            for batch_idx, (noisy_spikes, target_spikes, clean_logstft, noisy_logstft, log_min, log_max, _, joint_mask) in enumerate(dataloader):
                noisy = noisy_spikes.permute(1, 0, 2).to(self.device)           # [T, B, F]
                target = target_spikes.to(self.device)                          # kullanılmayabilir
                clean_logstft = clean_logstft.to(self.device)                  # [B, T, F]
                noisy_logstft = noisy_logstft.to(self.device)                  # [B, T, F]
                mask = joint_mask.to(self.device)

                # normalize geri çevir (isteğe bağlı)
                if self.cfg.normalize and self.predict_filter:
                    log_min = log_min.view(-1, 1, 1).to(self.device)
                    log_max = log_max.view(-1, 1, 1).to(self.device)
                    clean_logstft = clean_logstft * (log_max - log_min) + log_min
                    noisy_logstft = noisy_logstft * (log_max - log_min) + log_min

                r_target = (noisy > 0).float().mean().item()
                self.logs['target_rate'].append(r_target)

                self.optimizer.zero_grad()
                self.net(noisy)

                _, rec = list(self.net.spk_rec.items())[-1]
                spike_out = rec.permute(1, 0, 2)  # [B, T, F]

                spike_out.register_hook(spike_hook)

                if self.predict_filter:
                    # Sadece maskeyi ver
                    mask_hat = spike_out.clamp(0.0, 1.0)  # [B, T, F]
                    loss = self.loss_fn(mask_hat, noisy_logstft, clean_logstft, mask)

                else:
                    # 2️⃣ Eski tarz doğrudan logSTFT tahmini (sanki target spike olmuş gibi)
                    loss = self.loss_fn(spike_out, target, mask=mask)

                self.logs['spike_loss'].append(loss.item())
                loss.backward()

                self._log_gradients()
                self._update_weights()

                spike_rate = (spike_out > 0).float().mean().item()
                self.logs['pred_rate'].append(spike_rate)

                logging.debug(
                    f"[Batch {batch_idx+1}/{total_batches}] SpikeRate: {spike_rate:.3f} | Loss: {loss.item():.3f}"
                )

                self.batch_losses.append(loss.item())
                total_loss += loss.item()

                eta = (time.time() - start_time) / (batch_idx + 1) * (total_batches - (batch_idx + 1))
                print(f"ETA for epoch: {eta:.1f}s ≈ {eta/60:.1f}min")

        return total_loss / len(dataloader)

    def _log_gradients(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                self.logs['gradient_norms'].append(param.grad.norm().item())
                #print(f"{name} grad norm: {param.grad.norm().item()}")

    def _update_weights(self):
        weight_before = {name: param.data.clone() for name, param in self.net.named_parameters() if param.requires_grad}
        self.optimizer.step()
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                delta = (param.data - weight_before[name]).norm().item()
                self.logs['layer_deltas'].setdefault(name, []).append(delta)
                #print(f"{name} change after step: {delta:.6f}")

    def evaluate(self, dataloader):
        self.net.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                if self.predict_filter:
                    # Dataları çek
                    noisy_spikes, _, clean_logstft, noisy_logstft, log_min, log_max, _, joint_mask = batch

                    # Cihaza al
                    noisy = noisy_spikes.permute(1, 0, 2).to(self.device)
                    clean_logstft = clean_logstft.to(self.device)
                    noisy_logstft = noisy_logstft.to(self.device)
                    joint_mask = joint_mask.to(self.device)
                    if self.cfg.normalize:
                        # log_min, log_max şu anda [B], önce [B,1,1] yapıyoruz:
                        log_min = log_min.view(-1, 1, 1).to(self.device)   # [B,1,1]
                        log_max = log_max.view(-1, 1, 1).to(self.device)   # [B,1,1]

                        # Artık broadcast uyacak:
                        clean_logstft = clean_logstft * (log_max - log_min) + log_min
                        noisy_logstft = noisy_logstft * (log_max - log_min) + log_min

                    # Modeli çalıştır
                    self.net(noisy)
                    _, rec = list(self.net.spk_rec.items())[-1]
                    mask_hat = rec.permute(1, 0, 2).clamp(0.0, 1.0)

                    # 🔧 Sadece maskeyi veriyoruz
                    loss = self.loss_fn(mask_hat, noisy_logstft, clean_logstft, joint_mask)

                else:
                    noisy_spikes, target_spikes, *_, log_min, log_max, _, joint_mask = batch

                    noisy = noisy_spikes.permute(1, 0, 2).to(self.device)
                    target = target_spikes.to(self.device)
                    mask = joint_mask.to(self.device)

                    self.net(noisy)
                    _, rec = list(self.net.spk_rec.items())[-1]
                    spike_out = rec.permute(1, 0, 2)

                    if self.encode_mode == 'delta':
                        loss = self.loss_fn(spike_out, target, log_min=log_min, log_max=log_max, mask=mask)
                    else:
                        loss = self.loss_fn(spike_out, target, mask=mask)

                total_loss += loss.item()
        return total_loss / len(dataloader)


    def log_epoch(self, epoch, train_loss, val_loss, start_time):
        elapsed = time.time() - start_time
        logging.info(
            f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Elapsed: {elapsed:.1f}s"
        )

    def save_model(self, out_dir, out_file="trained_state_dict.pt"):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, out_file)
        torch.save(self.net.state_dict(), path)
        logging.info(f"Model state dict saved to {path}")

    def save_logs(self, out_file="logs.json", out_dir="checkpoints"):
        import json
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, out_file)
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=2)
        logging.info(f"Training logs saved to {path}")

    def save_checkpoint(self, path=None, epoch=0):
        model_type = getattr(self.net.cfg, "model_type", "unknown")  # <-- cfg içinden çek
        out_dir = os.path.join("checkpoints", model_type)             # <-- klasörü modele göre ayarla
        os.makedirs(out_dir, exist_ok=True)
        
        if path is None:
            path = os.path.join(out_dir, f"checkpoint_e{epoch}.pth")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'logs': self.logs
        }, path)
        
        logging.info(f"Checkpoint saved to {path}")



    def resume_from_checkpoint(self, path=None):
        model_type = getattr(self.net.cfg, "model_type", "unknown")
        out_dir = os.path.join("checkpoints", model_type)

        if path is None:
            # En son checkpoint dosyasını bul
            checkpoints = [f for f in os.listdir(out_dir) if f.endswith(".pth")]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {out_dir}")
            latest_checkpoint = sorted(checkpoints)[-1]
            path = os.path.join(out_dir, latest_checkpoint)

        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.logs = checkpoint['logs']
        logging.info(f"Resumed from checkpoint {path}")
        return checkpoint['epoch'] + 1

