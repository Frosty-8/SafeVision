"""
SafeVision AI

Radar Feature Encoder

Encodes radar feature maps into the
shared embedding space.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.utils.logger import logger


class ResidualBlock(nn.Module):
    """
    Lightweight residual block.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels,
            ),
            nn.ReLU(
                inplace=True,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels,
            ),
        )

        self.activation = nn.ReLU(
            inplace=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        out = self.layers(x)

        out += identity

        return self.activation(out)


class RadarEncoder(nn.Module):
    """
    Radar feature encoder.

    Converts radar tensors into the
    common embedding space.
    """

    def __init__(
        self,
        input_channels: int = 5,
        embedding_dim: int = 256,
    ) -> None:

        super().__init__()

        self.input_channels = input_channels

        self.embedding_dim = embedding_dim

        self.encoder = self._build_encoder()

        self.projection = self._build_projection()

        logger.info("RadarEncoder initialized.")

    def _build_encoder(
        self,
    ) -> nn.Module:
        """
        Build radar encoder.
        """

        return nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(
                128,
            ),
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def _build_projection(
        self,
    ) -> nn.Module:
        """
        Project radar features into the
        common embedding space.
        """

        return nn.Sequential(
            nn.Conv2d(
                256,
                self.embedding_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.embedding_dim,
            ),
            nn.ReLU(
                inplace=True,
            ),
        )

    def forward(
        self,
        radar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode radar features.

        Parameters
        ----------
        radar:
            Tensor of shape
            (B,C,H,W)

        Returns
        -------
        Tensor
            (B,256,H,W)
        """

        features = self.encoder(
            radar,
        )

        embeddings = self.projection(
            features,
        )

        return embeddings

    @property
    def output_channels(
        self,
    ) -> int:
        """
        Output embedding dimension.
        """

        return self.embedding_dim
