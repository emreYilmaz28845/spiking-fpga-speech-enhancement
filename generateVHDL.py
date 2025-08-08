from spikerplus import VhdlGenerator, Optimizer
from spikerplus.vhdl import write_vhdl
from models.builder import build_network, get_network_dict
from data.dataloader import get_dataloaders
from utils.config import cfg
import torch
import math

import logging
logging.basicConfig(level=logging.INFO) 

path = "Trained/2025-07-26_22-29_phased_rate_e60_len4000_arch_spiking-fsb-conv/trained_state_dict.pt"
# print(get_network_dict(cfg))
snn = build_network(cfg)
snn.load_state_dict(torch.load(path))

new_config = {
    "weights_bw" : 6,
    "neurons_bw" : 16,  
    "fp_dec" : 7 
}

vhdl_generator = VhdlGenerator(snn, new_config)

vhdl_snn  = vhdl_generator.generate(functional = False, interface = False)

write_vhdl(vhdl_snn, "network_VHDL")

