import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import csv
import torchaudio
import torch
from tqdm import tqdm

from utils.encode import spike_encode, reconstruct_from_spikes
from utils.audio_utils import reconstruct_without_stretch
from data.dataloader_forRecon import SpikeSpeechEnhancementDatasetForRecon

# === Global parameters
sample_rate = 16000
n_iter = 32
normalize_flag = True
padding = True

# === Hyperparameter grid (mode-specific hop_lengths)
mode_to_hop_lengths = {
    #"delta": [256],
    "phased_rate": [16, 32, 64, 128]  
    #"sod": [256],
    #"rate": [256],
    #"basic": [256]
}
encode_modes = ["phased_rate"]#, "phased_rate", "sod", "rate", "basic"]
n_ffts = [512]

# === Paths
data_root = r"C:/VSProjects/spiking-fpga-project/audio"
input_dir = os.path.join(data_root, "clean_16000")

# === Output: Runtime log CSV
log_file = "reconstruction_runtime_log.csv"
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["encode_mode", "hop_length", "n_fft", "max_len", "duration_sec"])

# === Grid loop
for encode_mode in encode_modes:
    hop_lengths = mode_to_hop_lengths[encode_mode]

    for hop_length in hop_lengths:
        for n_fft in n_ffts:
            print(f"\n=== Running for mode={encode_mode}, hop={hop_length}, n_fft={n_fft} ===")

            # Set encoding threshold
            if encode_mode == "sod":
                encode_threshold = 0.02
            elif encode_mode == "delta":
                encode_threshold = 0.003
            elif encode_mode == "basic":
                encode_threshold = 0.5
            else:
                encode_threshold = 0.003

            # === Build dataset
            dataset = SpikeSpeechEnhancementDatasetForRecon(
                clean_dir=input_dir,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                threshold=encode_threshold,
                mode=encode_mode,
                max_len=None,
                padding=padding,
                normalize=normalize_flag
            )

            max_len = dataset.max_len
            output_dir = f"{data_root}/compare_recon/{encode_mode}_Hop={hop_length}_Length={max_len}_NFFT={n_fft}"
            os.makedirs(output_dir, exist_ok=True)

            # === Start timing
            start_time = time.time()

            for i in tqdm(range(len(dataset)), desc=f"{encode_mode}, hop={hop_length}, fft={n_fft}"):
                try:
                    target_spikes, clean_logstft, log_min, log_max, original_length, _ = dataset[i]
                    target_reconstructed = reconstruct_from_spikes(target_spikes, mode=encode_mode)
                    ground_truth_vis = target_reconstructed.cpu().T  # [F, T]

                    output_path = os.path.join(output_dir, f"reconstructed_{i}.wav")

                    _ = reconstruct_without_stretch(
                        logstft_tensor=ground_truth_vis,
                        log_min=log_min,
                        log_max=log_max,
                        filename=output_path,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        sample_rate=sample_rate,
                        n_iter=n_iter,
                        original_length=original_length
                    )

                except Exception as e:
                    print(f"❌ Error reconstructing file {i}: {e}")

            # === End timing
            total_time = time.time() - start_time
            print(f"✅ Finished: {encode_mode}, hop={hop_length}, n_fft={n_fft} in {total_time:.2f} sec")

            # === Log timing
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([encode_mode, hop_length, n_fft, max_len, round(total_time, 2)])
