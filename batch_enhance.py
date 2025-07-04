import torch
import os
from tqdm import tqdm
from utils.encode import reconstruct_from_spikes
from utils.audio_utils import reconstruct_without_stretch
from models.builder import build_network
from data.dataloader import get_dataloaders
from utils.config import cfg
from itertools import chain

# === Ayarlar ===
noisy_dir = r"E:/VSProjects/datasets/audioVCTK/noisy"
out_dir = r"E:/VSProjects/datasets/audioVCTK/enhanced-Spiker"
os.makedirs(out_dir, exist_ok=True)

# === Model yükle ===
model_path = "Trained/2025-06-04_01-18_phased_rate_e1_len10000/trained_state_dict.pt"
snn = build_network(cfg)
snn.load_state_dict(torch.load(model_path))
snn.eval()

# === Dataset hazırla ===
cfg.data_root = r"E:/VSProjects/datasets/audioVCTK"  # clean klasörü de burada olmalı (kullanılmasa da)
train_loader, val_loader = get_dataloaders(cfg,shuffle=False)  # Sabit bölme için shuffle=False

print(f"Model: [train_loader, val_loader] = {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
global_index = 0  # tüm dataset boyunca index

for loader in [train_loader, val_loader]:
    for batch in tqdm(loader):
        input_spikes, _, _, _, log_min, log_max, original_length, mask = batch
        input_spikes = input_spikes.permute(1, 0, 2)  # [T, B, F]

        with torch.no_grad():
            snn(input_spikes)
            _, spike_out = list(snn.spk_rec.items())[-1]  # [T, B, F]
            spike_out = spike_out.permute(1, 0, 2)        # [B, T, F]

        for i in range(spike_out.shape[0]):
            T_real = int(mask[i].sum().item())
            trimmed = spike_out[i][:T_real]
            pred_logstft = reconstruct_from_spikes(trimmed, mode=cfg.encode_mode, trim=True)

            out_name = f"enhanced_{global_index:04d}.wav"
            reconstruct_without_stretch(
                pred_logstft.T,
                log_min[i].item(),
                log_max[i].item(),
                os.path.join(out_dir, out_name),
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                sample_rate=cfg.sample_rate,
                n_iter=cfg.n_iter
            )
            global_index += 1  


print(f"Tüm enhanced dosyalar: {out_dir}")
