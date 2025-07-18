import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

def plot_firing_rates(logs, save_path):
    batches = list(range(len(logs['target_rate'])))
    plt.figure(figsize=(8, 4))
    plt.plot(batches, logs['target_rate'], label='Target Rate')
    plt.plot(batches, logs['pred_rate'], label='Predicted Rate')
    plt.xlabel("Batch")
    plt.ylabel("Firing Rate")
    plt.title("Firing Rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

def plot_loss(logs, save_path):
    batches = list(range(len(logs['spike_loss'])))
    plt.figure(figsize=(8, 4))
    plt.plot(batches, logs['spike_loss'], label='Spike Loss')
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Spike Loss Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

# def plot_weight_deltas(logs, save_path):
#     batches = list(range(len(logs['spike_loss'])))
#     plt.figure(figsize=(10, 6))
#     for name, deltas in logs['layer_deltas'].items():
#         plt.plot(batches, deltas, label=f'‖ΔW‖₂ ({name})', linestyle='--', alpha=0.8)

#     plt.ylabel('‖ΔW‖₂')
#     plt.xlabel('Batch')
#     plt.title('Weight Changes Across Layers')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(save_path) 


def plot_weight_deltas(logs, save_path):
    batches = list(range(len(logs['spike_loss'])))
    plt.figure(figsize=(10, 6))

    # Use tab20 colormap for more distinct colors
    color_map = cm.get_cmap('tab20', len(logs['layer_deltas']))
    for idx, (name, deltas) in enumerate(logs['layer_deltas'].items()):
        plt.plot(batches, deltas, label=f'‖ΔW‖₂ ({name})',
                 linestyle='--', alpha=0.85, color=color_map(idx))

    plt.ylabel('‖ΔW‖₂')
    plt.xlabel('Batch')
    plt.title('Weight Changes Across Layers')
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()    

def plot_gradients(logs, save_path):
    batches = list(range(len(logs['gradient_norms'])))
    plt.figure(figsize=(8, 4))
    plt.plot(batches, logs['gradient_norms'], label='Gradient Norms')
    plt.xlabel("Batch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

def plot_learning_rate(logs, save_path):
    batches = list(range(len(logs['learning_rate'])))
    plt.figure(figsize=(8, 4))
    plt.plot(batches, logs['learning_rate'], label='Learning Rate')
    plt.xlabel("Batch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    
def plot_stft_comparison(output_dir, pred_spikes, target_spikes_vis,
                          predicted_vis, ground_truth_vis,
                          clean_logstft_vis, snn):
    # Create folder
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Spike and STFT comparison
    plt.figure(figsize=(14, 12))

    plt.subplot(3, 2, 1)
    plt.imshow(pred_spikes, aspect='auto', origin='lower')
    plt.title("Predicted Spikes")
    plt.xlabel("Time")
    plt.ylabel("STFT Bin")

    plt.subplot(3, 2, 2)
    plt.imshow(target_spikes_vis, aspect='auto', origin='lower')
    plt.title("Target Spikes")
    plt.xlabel("Time")
    plt.ylabel("STFT Bin")

    plt.subplot(3, 2, 3)
    plt.imshow(predicted_vis, aspect='auto', origin='lower')
    plt.title("Reconstructed Log-STFT (Predicted)")
    plt.xlabel("Time")
    plt.ylabel("STFT Bin")

    plt.subplot(3, 2, 4)
    plt.imshow(ground_truth_vis, aspect='auto', origin='lower')
    plt.title("Reconstructed Log-STFT (Target)")
    plt.xlabel("Time")
    plt.ylabel("STFT Bin")

    plt.subplot(3, 2, 5)
    plt.imshow(clean_logstft_vis, aspect='auto', origin='lower')
    plt.title("Original Clean Log-STFT")
    plt.xlabel("Time")
    plt.ylabel("STFT Bin")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "spike_logstft_comparison.png"))
    plt.close()

    # 2. Spiking activity for each layer
    layer_names = list(snn.spk_rec.keys())
    n_layers = len(layer_names)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 2 * n_layers), sharex=True)

    if n_layers == 1:
        axes = [axes]

    for i, (layer_name, rec) in enumerate(snn.spk_rec.items()):
        spikes = rec.permute(1, 0, 2)[0].cpu().numpy().T  # [N, T]
        axes[i].imshow(spikes, aspect='auto', origin='lower', interpolation='none')
        axes[i].set_ylabel(layer_name, rotation=0, labelpad=40)
        axes[i].yaxis.set_label_position("right")
        axes[i].grid(False)

    axes[-1].set_xlabel("Time step")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "spiking_activity_layers.png"))
    plt.close()
