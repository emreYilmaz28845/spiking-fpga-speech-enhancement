import torch.nn as nn



# def build_cnn(n_freq):
#     return nn.Sequential(
#         nn.Conv1d(n_freq, 512, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.Conv1d(512, 256, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.Conv1d(256, n_freq, kernel_size=1)
#     )


import torch.nn as nn
#moe snn_compatible
# def build_cnn(n_freq: int) -> nn.Module:
#     """Tamamı k=1 Conv1d ≡ Linear; SNN’e doğrudan kopyalanabilir."""
#     return nn.Sequential(
#         nn.Conv1d(n_freq, 512, kernel_size=1),  # = fc1
#         nn.GELU(),                              # non-linearity   (ReLU yerine daha yumuşak)
#         nn.Conv1d(512, 256, kernel_size=1),     # = fc2
#         nn.GELU(),
#         nn.Conv1d(256, n_freq, kernel_size=1)   # = fc3 (çıktı projeksiyonu)
#     )

def build_cnn(n_freq):
    return nn.Sequential(
        nn.Conv1d(n_freq, 256, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(256, 512, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(512, 1024, kernel_size=1,),
        nn.ReLU(),
        nn.Conv1d(1024, 512, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(512, 256, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(256, n_freq, kernel_size=1)
    )


