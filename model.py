from spikerplus import NetBuilder

def build_snn(n_inputs=40, n_outputs=40, n_cycles=73):
    """
    Builds a deeper SNN model for speech enhancement with dynamic LIF layers.
    Make sure spikerplus is configured to allow backprop.
    """
    net_dict = {
        "n_cycles": n_cycles,
        "n_inputs": n_inputs,

        "layer_0": {
            "neuron_model": "lif",
            "n_neurons": 128,  
            "beta": 0.9375,
            "threshold": 1.0,
            "reset_mechanism": "subtract"
        },
        "layer_1": {
            "neuron_model": "lif",
            "n_neurons": 256,  
            "beta": 0.9375,
            "threshold": 1.0,
            "reset_mechanism": "subtract"
        },
        "layer_2": {
            "neuron_model": "lif",
            "n_neurons": 512,  
            "beta": 0.9375,
            "threshold": 1.0,
            "reset_mechanism": "subtract"
        },
        "layer_3": {
            "neuron_model": "lif",
            "n_neurons": 256,  
            "beta": 0.9375,
            "threshold": 1.0,
            "reset_mechanism": "subtract"
        },
        "layer_4": {
            "neuron_model": "lif",
            "n_neurons": 128,  
            "beta": 0.9375,
            "threshold": 1.0,
            "reset_mechanism": "subtract"
        },
        "layer_5": {
            "neuron_model": "lif",
            "n_neurons": n_outputs,  
            "beta": 0.9375,
            "threshold": 1.0,
            "reset_mechanism": "none"
        }
    }

    net_builder = NetBuilder(net_dict)
    return net_builder.build()
