import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding3D(nn.Module):
    """Sinusoidal positional embedding for 3D coordinates.

    The embedding is generated independently for x, y and z directions and then
    summed.  This module returns a tensor of shape ``(x*y*z, embed_dim)`` which
    can be added to a sequence of flattened 3-D tokens.
    """

    def __init__(self, embed_dim: int, x_size: int, y_size: int, z_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        pe = self._build_pe()
        # register as buffer so it is moved to the correct device with the model
        self.register_buffer("pe", pe)

    def _build_pe(self) -> torch.Tensor:
        def _pos_enc(size: int, dim: int) -> torch.Tensor:
            position = torch.arange(size).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)
            )
            pe = torch.zeros(size, dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe

        dim_each = self.embed_dim // 3
        rem = self.embed_dim - 2 * dim_each
        px = _pos_enc(self.x_size, dim_each)
        py = _pos_enc(self.y_size, dim_each)
        pz = _pos_enc(self.z_size, rem)

        px = px[:, None, None, :]
        py = py[None, :, None, :]
        pz = pz[None, None, :, :]
        pe = px + py + pz
        pe = pe.view(self.x_size * self.y_size * self.z_size, self.embed_dim)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embedding to ``x``.

        Args:
            x: Tensor of shape ``(batch, seq_len, embed_dim)`` where
               ``seq_len == x_size * y_size * z_size``.
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class TransformerModel(nn.Module):
    """Transformer based sequence model with 3D positional encodings.

    The model expects an input tensor of shape ``(batch, channels, x, y)`` where
    ``channels`` corresponds to the number of input features per spatial
    location.  The ``x`` dimension is treated as the temporal axis while ``y``
    and ``z`` represent spatial axes.
    """

    def __init__(
        self,
        in_channels: int = 2,
        x_size: int = 20,
        y_size: int = 1,
        z_size: int = 1,
        embed_dim: int = 64,
        num_heads: int = 8,
        depth: int = 2,
        num_classes: int = 64,
    ):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.seq_len = x_size * y_size * z_size

        self.input_proj = nn.Linear(in_channels, embed_dim)
        self.pos_encoder = PositionalEncoding3D(embed_dim, x_size, y_size, z_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C, X, Y)
        b = x.size(0)
        x = x.permute(0, 2, 3, 1)  # (batch, X, Y, C)
        x = x.reshape(b, self.seq_len, -1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        x = x.view(b, self.x_size, self.y_size, -1)
        # return (batch, seq_len, num_classes)
        return x.squeeze(2)
