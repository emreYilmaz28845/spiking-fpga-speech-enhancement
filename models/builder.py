import torch
from spikerplus import NetBuilder


def build_network(cfg):
    net_dict = get_network_dict(cfg)
    net = NetBuilder(net_dict).build()

    # if getattr(cfg, "load_cnn_weights", False):
    #     from models.cnn import build_cnn
    #     cnn = build_cnn(cfg.n_freq_bins)
    #     checkpoint = torch.load("checkpoints/CNN/checkpoint_epoch_20.pth")
    #     cnn.load_state_dict(checkpoint["model_state_dict"])
    #     cnn.eval()

    #     with torch.no_grad():
    #         fc_idx = 1
    #         for layer in cnn:
    #             if isinstance(layer, torch.nn.Conv1d):
    #                 fc_name = f"fc{fc_idx}"
    #                 if fc_name in net.layers:
    #                     net.layers[fc_name].weight.data.copy_(layer.weight.squeeze(-1))
    #                     if net.layers[fc_name].bias is not None:
    #                         net.layers[fc_name].bias.data.copy_(layer.bias)
    #                     fc_idx += 1
    #                 else:
    #                     print(f"Warning: {fc_name} not in SNN layers")

    #     print(f"Successfully transferred {fc_idx - 1} Conv1d layers to SNN fc layers.")

    return net



def get_network_dict(cfg):
    example_input = torch.zeros(cfg.max_len, cfg.n_freq_bins)
    spike_threshold = cfg.spike_threshold
    model_type = getattr(cfg, "model_type", "homebrew")  # varsayılan: fullsubnet-tarzı

    if model_type == "homebrew":
        return {
            "n_cycles": example_input.shape[0],
            "n_inputs": example_input.shape[1],
            "layer_0": {
                "neuron_model": "syn", "n_neurons": 256,
                "alpha": 0.05, "learn_alpha": False,
                "beta": 0.05, "learn_beta": False,
                "threshold": spike_threshold + 0.6, "learn_threshold": False,
                "reset_mechanism": "zero"
            },
            "layer_1": {
                "neuron_model": "rsyn", "n_neurons": 128,
                "alpha": 0.07, "learn_alpha": False,
                "beta": 0.07, "learn_beta": False,
                "threshold": spike_threshold + 0.1, "learn_threshold": False,
                "reset_mechanism": "zero", "bias": True
            },
            "layer_2": {
                "neuron_model": "rif", "n_neurons": 64,
                "beta": 0.4, "learn_beta": True,
                "threshold": spike_threshold + 0.05, "learn_threshold": True,
                "reset_mechanism": "zero", "bias": True
            },
            "layer_3": {
                "neuron_model": "if", "n_neurons": 128,
                "alpha": 0.1, "learn_alpha": True,
                "beta": 0.6, "learn_beta": True,
                "threshold": spike_threshold, "learn_threshold": True,
                "reset_mechanism": "subtract", "bias": True
            },
            "layer_4": {
                "neuron_model": "if", "n_neurons": cfg.n_freq_bins,
                "threshold": spike_threshold * 0.3, "learn_threshold": False,
                "reset_mechanism": "zero", "bias": False
            },
        }

    elif model_type == "autoencoder":
        return {
            "n_cycles": example_input.shape[0],
            "n_inputs": example_input.shape[1],
            "layer_0": {
                "neuron_model": "lif", "n_neurons": 128,
                "threshold": spike_threshold, "learn_threshold": True,
                "reset_mechanism": "subtract", "bias": True
            },
            "layer_1": {
                "neuron_model": "lif", "n_neurons": 64,
                "threshold": spike_threshold + 0.1, "learn_threshold": True,
                "reset_mechanism": "subtract", "bias": True
            },
            "layer_2": {
                "neuron_model": "lif", "n_neurons": 128,
                "threshold": spike_threshold, "learn_threshold": True,
                "reset_mechanism": "subtract", "bias": True
            },
            "layer_3": {
                "neuron_model": "lif", "n_neurons": cfg.n_freq_bins,
                "threshold": spike_threshold * 0.5, "learn_threshold": False,
                "reset_mechanism": "zero", "bias": False
            }
        }
    elif model_type == "homebrew-delta":
        return {
            "n_cycles": example_input.shape[0],
            "n_inputs": example_input.shape[1],
            "layer_0": {
                "neuron_model": "syn", "n_neurons": 256,
                "alpha": 0.05, "learn_alpha": False,
                "beta": 0.05, "learn_beta": False,
                "threshold": spike_threshold+0.1, "learn_threshold": False,
                "reset_mechanism": "zero"
            },
            "layer_1": {
                "neuron_model": "rsyn", "n_neurons": 128,
                "alpha": 0.07, "learn_alpha": False,
                "beta": 0.07, "learn_beta": False,
                "threshold": spike_threshold, "learn_threshold": False,
                "reset_mechanism": "zero", "bias": True
            },
            "layer_2": {
                "neuron_model": "rif", "n_neurons": 64,
                "beta": 0.4, "learn_beta": True,
                "threshold": spike_threshold, "learn_threshold": True,
                "reset_mechanism": "zero", "bias": True
            },
            "layer_3": {
                "neuron_model": "if", "n_neurons": 128,
                "alpha": 0.1, "learn_alpha": True,
                "beta": 0.6, "learn_beta": True,
                "threshold": spike_threshold, "learn_threshold": True,
                "reset_mechanism": "zero", "bias": True
            },
            "layer_4": {
                "neuron_model": "if", "n_neurons": cfg.n_freq_bins,
                "threshold": spike_threshold * 0.3, "learn_threshold": False,
                "reset_mechanism": "subtract", "bias": False
            },
        }
    elif model_type == "dpsnn":
        return {
            "n_cycles": example_input.shape[0],
            "n_inputs": example_input.shape[1],
            "layer_0": {
                "neuron_model": "syn", "n_neurons": 256,
                "alpha": 0.05, "learn_alpha": False,
                "beta": 0.05, "learn_beta": False,
                "threshold": spike_threshold + 0.5, "learn_threshold": False,
                "reset_mechanism": "zero"
            },
            # "layer_1": {
            #     "neuron_model": "lif", "n_neurons": 256,
            #     "threshold": spike_threshold + 0.1, "learn_threshold": True,
            #     "reset_mechanism": "subtract", "bias": True
            # },
            # "layer_2": {
            #     "neuron_model": "lif", "n_neurons": 256,
            #     "threshold": spike_threshold, "learn_threshold": True,
            #     "reset_mechanism": "subtract", "bias": True
            # },
            "layer_1": {
                "neuron_model": "lif", "n_neurons": cfg.n_freq_bins,
                "threshold": spike_threshold * 0.5, "learn_threshold": False,
                "reset_mechanism": "zero", "bias": False
            }
        }
    elif model_type == "dpsnn_2":
        return {
            "n_cycles": example_input.shape[0],
            "n_inputs": example_input.shape[1],
            "layer_0": {
                "neuron_model": "syn", "n_neurons": 256,
                "alpha": 0.05, "learn_alpha": False,
                "beta": 0.05, "learn_beta": False,
                "threshold": spike_threshold + 0.5, "learn_threshold": False,
                "reset_mechanism": "zero"
            },
            "layer_2": {
                "neuron_model": "lif", "n_neurons": 256,
                "threshold": spike_threshold, "learn_threshold": True,
                "reset_mechanism": "subtract", "bias": True
            },
            "layer_1": {
                "neuron_model": "lif", "n_neurons": cfg.n_freq_bins,
                "threshold": spike_threshold * 0.5, "learn_threshold": False,
                "reset_mechanism": "zero", "bias": False
            }
        }
    elif model_type == "spiking-fsb":
        return {
            "n_cycles": example_input.shape[0],
            "n_inputs": example_input.shape[1],
            "layer_0": {
                "neuron_model": "rsyn",              
                "n_neurons": cfg.n_freq_bins,
                "alpha": 0.05,                       
                "learn_alpha": True,                 
                "beta": 0.05,                        
                "learn_beta": True,                  
                "threshold": spike_threshold + 0.5,  
                "learn_threshold": True,
                "reset_mechanism": "subtract"        
            },
            "layer_1": {
                "neuron_model": "rsyn",              
                "n_neurons": cfg.n_freq_bins,
                "alpha": 0.05,                       
                "learn_alpha": True,                 
                "beta": 0.05,                        
                "learn_beta": True,                  
                "threshold": spike_threshold,  
                "learn_threshold": True,
                "reset_mechanism": "subtract"        
            }
        }      
    elif model_type == "spiking-fsb-conv":
        return {
            "n_cycles": example_input.shape[0],
            "n_inputs": example_input.shape[1],
            "layer_0": {
                "neuron_model": "rsyn",               
                "n_neurons": 256,
                "alpha": 0.05,                        
                "learn_alpha": True,                 
                "beta": 0.05,                         
                "learn_beta": True,                  
                "threshold": spike_threshold + 0.2,  #phased_rate için 0.5, delta için 0.1
                "learn_threshold": True,
                "reset_mechanism": "subtract"        
            },
            "layer_1": {
                "neuron_model": "rsyn",               
                "n_neurons": 256,
                "alpha": 0.05,                        
                "learn_alpha": True,                 
                "beta": 0.05,                         
                "learn_beta": True,                  
                "threshold": spike_threshold + 0.2, #phased_rate için 0.5, delta için 0.1
                "learn_threshold": True,
                "reset_mechanism": "subtract"        
            },
            "layer_2": {
                "neuron_model": "lif", "n_neurons": cfg.n_freq_bins,
                "threshold": spike_threshold + 0.1, "learn_threshold": True, #phased_rate için 0.1, delta için 0
                "reset_mechanism": "subtract", "bias": True
            }
        }
    elif model_type == "spiking-fsb-conv-filter-predict":
        return {
            "n_cycles": example_input.shape[0],
            "n_inputs": example_input.shape[1],
            "layer_0": {
                "neuron_model": "rsyn",               
                "n_neurons": 256,
                "alpha": 0.05,                        
                "learn_alpha": True,                 
                "beta": 0.05,                         
                "learn_beta": True,                  
                "threshold": spike_threshold + 0.3,  #phased_rate için 0.5, delta için 0.1
                "learn_threshold": True,
                "reset_mechanism": "subtract"        
            },
            "layer_1": {
                "neuron_model": "rsyn",               
                "n_neurons": 256,
                "alpha": 0.05,                        
                "learn_alpha": True,                 
                "beta": 0.05,                         
                "learn_beta": True,                  
                "threshold": spike_threshold + 0.3, #phased_rate için 0.5, delta için 0.1
                "learn_threshold": True,
                "reset_mechanism": "subtract"        
            },
            "layer_2": {
                "neuron_model": "lif", "n_neurons": cfg.n_freq_bins,
                "threshold": spike_threshold + 0.2, "learn_threshold": True, #phased_rate için 0.1, delta için 0
                "reset_mechanism": "subtract", "bias": True
            }
        }
    elif model_type == "spiking-fsb-conv-noLif":
        return {
            "n_cycles": example_input.shape[0],
            "n_inputs": example_input.shape[1],
            "layer_0": {
                "neuron_model": "rsyn",               
                "n_neurons": 256,
                "alpha": 0.05,                        
                "learn_alpha": True,                 
                "beta": 0.05,                         
                "learn_beta": True,                  
                "threshold": spike_threshold + 0.1,  #phased_rate için 0.5, delta için 0.1
                "learn_threshold": True,
                "reset_mechanism": "subtract"        
            },
            "layer_1": {
                "neuron_model": "rsyn",               
                "n_neurons": cfg.n_freq_bins,
                "alpha": 0.05,                        
                "learn_alpha": True,                 
                "beta": 0.05,                         
                "learn_beta": True,                  
                "threshold": spike_threshold + 0.1, #phased_rate için 0.5, delta için 0.1
                "learn_threshold": True,
                "reset_mechanism": "subtract"        
            }
        }
    elif model_type == "spiking-fsb-conv-noRsyn2":
        return {
            "n_cycles": example_input.shape[0],
            "n_inputs": example_input.shape[1],
            "layer_0": {
                "neuron_model": "rsyn",               
                "n_neurons": 256,
                "alpha": 0.05,                        
                "learn_alpha": True,                 
                "beta": 0.05,                         
                "learn_beta": True,                  
                "threshold": spike_threshold + 0.1,  #phased_rate için 0.5, delta için 0.1
                "learn_threshold": True,
                "reset_mechanism": "subtract"        
            },
            "layer_1": {
                "neuron_model": "lif", "n_neurons": cfg.n_freq_bins,
                "threshold": spike_threshold, "learn_threshold": True, #phased_rate için 0.1, delta için 0
                "reset_mechanism": "subtract", "bias": False
            }
        }
    elif model_type == "CNNLike":
        return {
            "n_cycles": example_input.shape[0],  # T
            "n_inputs": example_input.shape[1],  # F

            # Layer 0: Temporal Smoothing (Conv1)
            "layer_0": {
                "neuron_model": "syn",
                "n_neurons": 256,
                "alpha": 0.1,
                "learn_alpha": True,
                "beta": 0.05,
                "learn_beta": True,
                "threshold": spike_threshold,
                "learn_threshold": True,
                "reset_mechanism": "subtract"
            },

            # Layer 1: Pattern Extraction (Conv2)
            "layer_1": {
                "neuron_model": "rsyn",
                "n_neurons": 256,
                "alpha": 0.05,
                "learn_alpha": True,
                "beta": 0.05,
                "learn_beta": True,
                "threshold": spike_threshold + 0.1,
                "learn_threshold": True,
                "reset_mechanism": "subtract"
            },

            # Layer 2: Temporal Memory (Conv3)
            "layer_2": {
                "neuron_model": "rsyn",
                "n_neurons": 128,
                "alpha": 0.03,
                "learn_alpha": True,
                "beta": 0.03,
                "learn_beta": True,
                "threshold": spike_threshold + 0.15,
                "learn_threshold": True,
                "reset_mechanism": "subtract"
            },

            # Layer 3: Output projection
            "layer_3": {
                "neuron_model": "lif",
                "n_neurons": example_input.shape[1], 
                "threshold": spike_threshold + 0.2,
                "learn_threshold": True,
                "reset_mechanism": "subtract"
            }
        },
    elif model_type == "cnn-to-snn":
        return {
            "n_cycles": example_input.shape[0],  # T
            "n_inputs": example_input.shape[1],  # F
            "layer_0": {
                "neuron_model": "lif", "n_neurons": 512,
                "threshold": spike_threshold + 0.1, "learn_threshold": False,
                "reset_mechanism": "subtract", "bias": True
            },
            "layer_1": {
                "neuron_model": "lif", "n_neurons": 256,
                "threshold": spike_threshold + 0.1, "learn_threshold": False,
                "reset_mechanism": "subtract", "bias": True
            },
            "layer_2": {
                "neuron_model": "lif", "n_neurons": cfg.n_freq_bins,
                "threshold": spike_threshold + 0.1, "learn_threshold": False,
                "reset_mechanism": "subtract", "bias": True
            }
        }
    elif model_type == "cnn-to-snn-2":
        return {
            "n_cycles": example_input.shape[0],   # T
            "n_inputs": example_input.shape[1],   # F (n_freq)
            
            "layer_0": {
                "neuron_model": "lif",            # veya rsyn / rlif vs.
                "n_neurons": 256,
                "threshold": spike_threshold,
                "reset_mechanism": "subtract"
            },
            "layer_1": {
                "neuron_model": "lif",
                "n_neurons": 512,
                "threshold": spike_threshold,
                "reset_mechanism": "subtract"
            },
            "layer_2": {
                "neuron_model": "lif",
                "n_neurons": 1024,
                "threshold": spike_threshold,
                "reset_mechanism": "subtract"
            },
            "layer_3": {
                "neuron_model": "lif",
                "n_neurons": 512,
                "threshold": spike_threshold,
                "reset_mechanism": "subtract"
            },
            "layer_4": {
                "neuron_model": "lif",
                "n_neurons": 256,
                "threshold": spike_threshold,
                "reset_mechanism": "subtract"
            },
            "layer_5": {
                "neuron_model": "lif",
                "n_neurons": example_input.shape[1],  # output same as input freq bins
                "threshold": spike_threshold,
                "reset_mechanism": "subtract"
            }
        }


    else:
        raise ValueError(f"Unknown model_type: {model_type}")
