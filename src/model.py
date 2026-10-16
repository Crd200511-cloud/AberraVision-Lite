"""Five-layer residual CNN: (observation, Wiener) -> correction + Wiener."""
import torch
from torch import nn


class ResidualCNN(nn.Module):
    receptive_radius = 5

    def __init__(self):
        super().__init__()
        channels = [2, 32, 32, 32, 32, 1]
        layers = []
        for i, (inputs, outputs) in enumerate(zip(channels[:-1], channels[1:])):
            layers.append(nn.Conv2d(inputs, outputs, kernel_size=3, padding=1))
            if i < 4:
                layers.append(nn.ReLU())
        self.correction = nn.Sequential(*layers)
        # Initial prediction is exactly the shared Wiener baseline.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 2:
            raise ValueError('Expected [batch, 2, height, width]: observation then Wiener.')
        return inputs[:, 1:2] + self.correction(inputs)
    
