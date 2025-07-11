import torch.nn as nn



def build_cnn(n_freq):
    return nn.Sequential(
        nn.Conv1d(n_freq, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(256, n_freq, kernel_size=1)
    )