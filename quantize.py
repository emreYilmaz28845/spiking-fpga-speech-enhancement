import torch

state_dict = torch.load("Trained/2025-08-06_08-54_phased_rate_e100_len3000_arch_spiking-fsb-conv-light/trained_state_dict.pt")
quantized_state_dict = {}

for key, value in state_dict.items():
    if torch.is_floating_point(value):
        quantized = torch.clamp(value, -1.0, 1.0)  # sınırlı aralığa getir
        quantized = (quantized * 127).round() / 127  # 8-bit'e benzet (int8)
        quantized_state_dict[key] = quantized
    else:
        quantized_state_dict[key] = value  # örneğin int olanlar bozulmasın

torch.save(quantized_state_dict, "trained_state_dict_int8.pt")
