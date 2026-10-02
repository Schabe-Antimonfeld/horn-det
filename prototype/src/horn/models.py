"""Document baseline and pending neural candidates; CPU PyTorch is optional."""

import torch
from torch import nn
from torch.nn import functional as functional


class Separable(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming, incoming, 3, padding=1, groups=incoming),
                                    nn.ReLU(), nn.Conv2d(incoming, outgoing, 1), nn.ReLU())

    def forward(self, values):
        return self.layers(values)


class TinyCNN(nn.Module):
    """Chapter 07 channels, engineering choice: stride one, same padding and ReLU."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                     Separable(16, 24), Separable(24, 32), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(32, 2)

    def forward(self, values):
        return self.head(self.encoder(values).flatten(1))


class SmallMLP(nn.Module):
    def __init__(self, features=26, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, values):
        return self.layers(values)


class CRNN(nn.Module):
    """Frequency-only convolutions plus a unidirectional GRU with explicit cached state."""

    def __init__(self, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, (3, 1), padding=(1, 0)), nn.ReLU(),
                                     nn.Conv2d(16, 24, (3, 1), padding=(1, 0)), nn.ReLU())
        self.recurrent = nn.GRU(24, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def sequence(self, values, state=None):
        features = self.encoder(values).mean(dim=2).transpose(1, 2)
        output, state = self.recurrent(features, state)
        return self.head(output), state

    def forward(self, values):
        return self.sequence(values)[0][:, -1]


def distillation_loss(student, teacher, labels, temperature=2.0, mixture=0.5):
    if temperature <= 0 or not 0 <= mixture <= 1:
        raise ValueError("Invalid distillation settings")
    hard = functional.cross_entropy(student, labels)
    soft = functional.kl_div(functional.log_softmax(student / temperature, dim=-1),
                             functional.softmax(teacher.detach() / temperature, dim=-1),
                             reduction="batchmean") * temperature ** 2
    return (1 - mixture) * hard + mixture * soft


def create_model(name):
    if name == "tiny":
        return TinyCNN()
    if name == "mlp":
        return SmallMLP()
    if name == "crnn":
        return CRNN()
    raise ValueError("Unknown model")
