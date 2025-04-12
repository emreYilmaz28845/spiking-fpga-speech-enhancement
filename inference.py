import torch
import torchaudio
import logging

from preprocess import load_and_encode_spikes
from reconstruct import decode_spikes_to_audio
from model import build_snn

logging.basicConfig(level=logging.INFO)

def main():
    input_wav = "audio/noisy/156.wav"
    output_wav = "audio/enhanced/enhanced.wav"
    model_path = "Trained/enhancement_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Encoding {input_wav} to spikes...")
    noisy_spikes, _ = load_and_encode_spikes(input_wav, return_mel=False)
    if noisy_spikes is None:
        raise ValueError(f"Could not process {input_wav}. Possibly silent or invalid.")
    noisy_spikes = noisy_spikes.to(device)  # shape [1, T, F]
    
    T, F = noisy_spikes.shape[1], noisy_spikes.shape[2]

    model = build_snn(n_inputs=F, n_outputs=40, n_cycles=T).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logging.info("Running inference...")
    with torch.no_grad():
        spikes_in = noisy_spikes.permute(1, 0, 2)  # => [T, B=1, F]
        _ = model(spikes_in)
        # Make sure the final layer key is correct
        out_spikes = model.spk_rec["lif6"].permute(1, 0, 2)  # => [B=1, T, F]

    logging.info("Decoding spikes to waveform...")
    enhanced_wav = decode_spikes_to_audio(out_spikes.cpu())  # shape [time]

    # Make sure parent dirs exist
    import os
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    torchaudio.save(output_wav, enhanced_wav.unsqueeze(0), sample_rate=16000)
    logging.info(f"Enhanced wav saved to: {output_wav}")

if __name__ == "__main__":
    main()
