import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from typing import Optional

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
    
def plot_stft_comparison(
    output_dir: str,
    pred_spikes: Optional[np.ndarray],
   target_spikes_vis: Optional[np.ndarray],
    predicted_vis: np.ndarray,
    ground_truth_vis: np.ndarray,
    clean_logstft_vis: np.ndarray,
    snn
):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1) Spike / STFT karşılaştırma
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    # A: Predicted Spikes
    if pred_spikes is not None:
        im0 = axes[0].imshow(pred_spikes, aspect='auto', origin='lower')
        axes[0].set_title("Predicted Spikes")
        fig.colorbar(im0, ax=axes[0])
    else:
        axes[0].axis('off')

    # B: Target Spikes
    if target_spikes_vis is not None:
        im1 = axes[1].imshow(target_spikes_vis, aspect='auto', origin='lower')
        axes[1].set_title("Target Spikes")
        fig.colorbar(im1, ax=axes[1])
    else:
        axes[1].axis('off')

    # C: Reconstructed Predicted STFT
    im2 = axes[2].imshow(predicted_vis, aspect='auto', origin='lower')
    axes[2].set_title("Reconstructed Log-STFT (Predicted)")
    fig.colorbar(im2, ax=axes[2])

    # D: Reconstructed Target STFT
    im3 = axes[3].imshow(ground_truth_vis, aspect='auto', origin='lower')
    axes[3].set_title("Reconstructed Log-STFT (Target)")
    fig.colorbar(im3, ax=axes[3])

    # E: Clean STFT
    im4 = axes[4].imshow(clean_logstft_vis, aspect='auto', origin='lower')
    axes[4].set_title("Original Clean Log-STFT")
    fig.colorbar(im4, ax=axes[4])

    # F: boş bırak
    axes[5].axis('off')

    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency Bin")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "spike_logstft_comparison.png"))
    plt.close(fig)

    # 2) Her layer için spiking activity
    layer_names = list(snn.spk_rec.keys())
    n_layers = len(layer_names)
    fig2, axs2 = plt.subplots(n_layers, 1, figsize=(12, 2 * n_layers), sharex=True)
    if n_layers == 1:
        axs2 = [axs2]

    for ax, layer_name in zip(axs2, layer_names):
        rec = snn.spk_rec[layer_name]          # [T, B, F]
        spikes = rec.permute(1,0,2)[0].cpu().numpy().T  # [F, T]
        im = ax.imshow(spikes, aspect='auto', origin='lower', interpolation='none')
        ax.set_ylabel(layer_name, rotation=0, labelpad=40)
        ax.yaxis.set_label_position("right")
        fig2.colorbar(im, ax=ax)

    axs2[-1].set_xlabel("Time Step")
    plt.tight_layout()
    fig2.savefig(os.path.join(plot_dir, "spiking_activity_layers.png"))
    plt.close(fig2)