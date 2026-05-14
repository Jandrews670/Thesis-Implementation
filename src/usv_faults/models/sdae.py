from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


class SparseDenoisingAutoencoder(nn.Module):
    """Configurable MLP autoencoder with a separable encoder and decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        hidden_activation: str = "relu",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(previous, hidden))
            encoder_layers.append(_activation(hidden_activation))
            previous = hidden
        encoder_layers.append(nn.Linear(previous, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        previous = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(previous, hidden))
            decoder_layers.append(_activation(hidden_activation))
            previous = hidden
        decoder_layers.append(nn.Linear(previous, input_dim))
        if output_activation != "linear":
            decoder_layers.append(_activation(output_activation))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "linear":
        return nn.Identity()
    raise ValueError(f"unsupported activation: {name}")
