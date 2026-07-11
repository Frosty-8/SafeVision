from __future__ import annotations

import math

import torch
import torch.nn as nn

from app.utils.logger import logger


class PositionalEncoding(nn.Module):
    """
    Two-dimensional sinusoidal
    positional encoding.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        temperature: int = 10000,
    ) -> None:

        super().__init__()

        if embedding_dim % 2 != 0:

            raise ValueError("Embedding dimension must be even.")

        self.embedding_dim = embedding_dim

        self.temperature = temperature

        logger.info("PositionalEncoding initialized.")

    def _build_encoding(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build 2D sinusoidal positional
        encoding.
        """

        y = torch.arange(
            height,
            device=device,
            dtype=torch.float32,
        )

        x = torch.arange(
            width,
            device=device,
            dtype=torch.float32,
        )

        y_grid, x_grid = torch.meshgrid(
            y,
            x,
            indexing="ij",
        )

        dim = self.embedding_dim // 4

        omega = torch.arange(
            dim,
            device=device,
            dtype=torch.float32,
        )

        omega = 1.0 / (self.temperature ** (omega / dim))

        x = x_grid[..., None] * omega

        y = y_grid[..., None] * omega

        position = torch.cat(
            (
                x.sin(),
                x.cos(),
                y.sin(),
                y.cos(),
            ),
            dim=-1,
        )

        return position.permute(
            2,
            0,
            1,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add positional encoding to features.

        Input
        -----
        (B,C,H,W)

        Output
        ------
        (B,C,H,W)
        """

        _, _, height, width = features.shape

        encoding = self._build_encoding(
            height,
            width,
            features.device,
        )

        return features + encoding.unsqueeze(0)
